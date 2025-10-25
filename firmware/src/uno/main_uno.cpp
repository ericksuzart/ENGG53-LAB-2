#include <ArduinoSTL.h>

#include <array>

#undef min
#undef max

#include <vt_kalman>
#include <vt_linalg>

// ============================================================================
// CONSTANTS AND CONFIGURATION
// ============================================================================

// Encoder has 8 white sections and 8 cutouts, producing 16 transitions per revolution
constexpr int ENCODER_COUNTS_PER_REVOLUTION = 16;
constexpr float RADIANS_PER_COUNT = 2.0F * PI / static_cast<float>(ENCODER_COUNTS_PER_REVOLUTION);
constexpr uint8_t MIN_SAMPLE_COUNT = 50U;  // Minimum samples for encoder threshold calibration

// Kalman filter dimensions
constexpr int STATE_DIM = 3;    // [angle, velocity, acceleration]
constexpr int MEAS_DIM = 1;     // Only angle measurement
constexpr int CONTROL_DIM = 3;  // [dt, pwm, direction]

// Serial communication
constexpr int SERIAL_BUFFER_SIZE = 32;

// Motor interface pins
constexpr int LEFT_PWM_PIN = 6;
constexpr int LEFT_DIR_PIN = 8;
constexpr int LEFT_ENC_PIN = 0;

constexpr int RIGHT_PWM_PIN = 5;
constexpr int RIGHT_DIR_PIN = 7;
constexpr int RIGHT_ENC_PIN = 1;

// ADC constants
constexpr uint8_t ADC_MAX_VALUE = 255U;
constexpr uint8_t ADC_INITIAL_THRESHOLD = 128U;
constexpr uint8_t PIN_MASK = 0x07U;
constexpr uint8_t MAX_PIN_GROUP_1 = 7U;
constexpr uint8_t MAX_PIN_GROUP_2 = 13U;
constexpr uint8_t MAX_PIN_GROUP_3 = 19U;

// Motor model constants
constexpr float PWM_TO_ACCELERATION_GAIN = 0.01F;
constexpr float FRICTION_COEFFICIENT = 0.003F;
constexpr float STATIC_FRICTION_THRESHOLD = 140.0F;
constexpr float STUCK_VELOCITY_THRESHOLD = 0.05F;
constexpr float MAX_ACCELERATION = 200.0F;
constexpr float MIN_ACCELERATION = -200.0F;
constexpr float BRAKING_COEFFICIENT = 10.0F;         // tune needed
constexpr float STATIC_FRICTION_COEFFICIENT = 5.0F;  // tune needed
constexpr float VELOCITY_THRESHOLD = 0.1F;

// Timing constants
constexpr float MIN_DELTA_TIME = 0.001F;
constexpr float MAX_DELTA_TIME = 0.1F;
constexpr float INITIAL_DELTA_TIME = 0.01F;
constexpr uint32_t INITIAL_DELAY_MS = 100U;
constexpr uint16_t LOOP_DELAY_MICROS = 100U;
constexpr uint32_t TELEMETRY_INTERVAL_MS = 50U;  // 20Hz
constexpr uint32_t STUCK_THRESHOLD_MS = 10000U;  // 10 seconds
constexpr uint32_t MICROSECONDS_PER_SECOND = 1000000UL;
constexpr uint32_t MILLISECONDS_PER_SECOND = 1000UL;

// Additional constants
constexpr float HALF = 0.5F;

// ============================================================================
// ALIASES
// ============================================================================

using EKF = vt::extended_kalman_filter_t<STATE_DIM, MEAS_DIM, CONTROL_DIM>;
using StateVector = vt::numeric_vector<STATE_DIM>;
using MeasurementVector = vt::numeric_vector<MEAS_DIM>;
using ControlVector = vt::numeric_vector<CONTROL_DIM>;

// ============================================================================
// GLOBAL VARIABLES
// ============================================================================

// --- Encoder ---

volatile uint8_t current_channel = 0U;  // ADC channel being read
volatile uint8_t left_raw_value = 0U;   // Latest ADC value for left encoder
volatile uint8_t right_raw_value = 0U;  // Latest ADC value for right encoder
volatile bool new_left_data = false;    // Flag for new left encoder data
volatile bool new_right_data = false;   // Flag for new right encoder data

// Tracking
int32_t left_enc_count = 0U;
int32_t right_enc_count = 0U;
uint8_t left_prev_state = 0;
uint8_t right_prev_state = 0;

// Dynamic threshold calibration
uint8_t left_min = ADC_MAX_VALUE;
uint8_t left_max = 0U;
uint8_t right_min = ADC_MAX_VALUE;
uint8_t right_max = 0U;
uint8_t left_threshold = ADC_INITIAL_THRESHOLD;
uint8_t right_threshold = ADC_INITIAL_THRESHOLD;
uint8_t left_sample_count = 0U;
uint8_t right_sample_count = 0U;

// --- Encoder ---

// --- Motor control ---

uint8_t left_pwm = 0U;
uint8_t right_pwm = 0U;
int8_t left_direction = 1;   // 1 for forward, -1 for backward
int8_t right_direction = 1;  // 1 for forward, -1 for backward

// --- Motor control ---

// Serial communication
std::array<char, SERIAL_BUFFER_SIZE> serialBuffer;
uint8_t serialIndex = 0U;
bool newCommand = false;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * @brief Normalizes angle to [-PI, PI] range
 */
inline float normalize_angle(float angle)
{
    float normalized = fmodf(angle, TWO_PI);
    if (normalized > PI)
        normalized -= TWO_PI;
    else if (normalized < -PI)
        normalized += TWO_PI;
    return normalized;
}

/**
 * @brief Non-blocking millisecond delay using timer comparison
 * @param last_time Pointer to last time reference
 * @param ms_delay Delay in milliseconds
 * @return true if delay completed, false otherwise
 */
bool nonBlockingDelay(uint32_t & last_time, const uint32_t & ms_delay)
{
    const uint32_t current_time = millis();
    if (current_time - last_time >= ms_delay)
    {
        last_time = current_time;
        return true;
    }
    return false;
}

// ============================================================================
// KALMAN FILTER IMPLEMENTATION
// ============================================================================

/**
 * @brief Kalman filter state management for a single wheel
 */
class WheelKalmanState
{
   private:
    StateVector state_vector_;
    EKF filter_;
    uint32_t last_update_us_ = 0U;
    bool initialized_ = false;

    // Continuous rotation tracking
    float wrapped_angle_offset_ = 0.0F;
    float previous_normalized_angle_ = 0.0F;
    bool angle_continuity_initialized_ = false;

    // Movement detection
    uint32_t last_encoder_count_ = 0U;
    uint32_t last_movement_time_ = 0U;
    bool was_moving_ = false;

    // Noise matrices - made const to fix warnings
    static const vt::numeric_matrix<STATE_DIM, STATE_DIM> process_noise_covariance_;
    static const vt::numeric_matrix<MEAS_DIM, MEAS_DIM> measurement_noise_covariance_;

   public:
    WheelKalmanState()
    : state_vector_(create_zero_vector()),
      filter_(state_transition_function, state_transition_jacobian, measurement_function, measurement_jacobian,
              process_noise_covariance_, measurement_noise_covariance_, state_vector_)
    {
    }

    // Getters
    const StateVector & get_state_vector() const { return filter_.state_vector; }
    bool is_initialized() const { return initialized_; }
    uint32_t get_last_update_us() const { return last_update_us_; }

    /**
     * @brief Creates a zero-initialized state vector
     */
    static StateVector create_zero_vector()
    {
        StateVector vector;

        for (unsigned int i = 0U; i < STATE_DIM; ++i)
            vector[i] = 0.0F;

        return vector;
    }

    /**
     * @brief State transition function
     */
    /**
     * @brief State transition function
     */
    static StateVector state_transition_function(const StateVector & prev_state, const ControlVector & control_input)
    {
        // Unpack control vector
        const float delta_time = control_input[0];
        const float pwm_command = control_input[1];
        const float direction_command = control_input[2];

        // Unpack previous state vector
        const float current_angle = prev_state[0];
        const float current_velocity = prev_state[1];
        const float current_acceleration = prev_state[2];  // Used for position integration

        const float half_delta_time_squared = HALF * delta_time * delta_time;

        // Is the motor being commanded to move above the static friction threshold?
        const bool is_commanded = pwm_command > STATIC_FRICTION_THRESHOLD;

        // Is the motor stuck (commanded to move but not moving)?
        const bool is_stuck = is_commanded && (fabsf(current_velocity) < STUCK_VELOCITY_THRESHOLD);

        // Is the command fighting the current velocity (braking)?
        const bool direction_mismatch = (current_velocity * direction_command) < 0.0F;

        float modeled_acceleration = 0.0F;

        if (is_commanded && !is_stuck)
        {
            // --- State 1: Normal Operation (Commanded and Moving) ---
            if (direction_mismatch && fabsf(current_velocity) > VELOCITY_THRESHOLD)
            {
                // Active braking for direction reversal
                modeled_acceleration = -current_velocity * BRAKING_COEFFICIENT;
            }
            else
            {
                // Normal operation: PWM force minus dynamic friction
                const float pwm_effect =
                  direction_command * PWM_TO_ACCELERATION_GAIN * (pwm_command - STATIC_FRICTION_THRESHOLD);
                modeled_acceleration = pwm_effect - current_velocity * FRICTION_COEFFICIENT;
            }
        }
        else if (is_stuck)
        {
            // --- State 2: Stuck (Commanded but Not Moving) ---
            // Apply strong static friction to model the "stuck" state
            modeled_acceleration = -current_velocity * STATIC_FRICTION_COEFFICIENT;
        }
        else  // !is_commanded
        {
            // --- State 3: Coasting (Not Commanded) ---
            // Apply normal dynamic friction and braking force
            modeled_acceleration = -current_velocity * (FRICTION_COEFFICIENT + 1.0F);
        }

        // Limit acceleration for numerical stability
        modeled_acceleration = constrain(modeled_acceleration, MIN_ACCELERATION, MAX_ACCELERATION);

        // --- 4. Predict Next State using Kinematic Equations ---
        StateVector next_state;

        // p_k = p_k-1 + v_k-1*dt + 0.5*a_k-1*dt^2
        next_state[0] = current_angle + delta_time * current_velocity + half_delta_time_squared * current_acceleration;

        // v_k = v_k-1 + a_k*dt (Uses the *new* modeled acceleration)
        next_state[1] = current_velocity + delta_time * modeled_acceleration;

        // a_k = a_k (The new acceleration is the one we just modeled)
        next_state[2] = modeled_acceleration;

        return next_state;
    }

    /**
     * @brief Jacobian of state transition function
     */
    static vt::numeric_matrix<STATE_DIM, STATE_DIM> state_transition_jacobian(const StateVector & state,
                                                                              const ControlVector & control)
    {
        const float delta_time = control[0];
        const float pwm_command = control[1];
        const float direction_command = control[2];
        const float current_velocity = state[1];
        const float signed_pwm = pwm_command * direction_command;

        vt::numeric_matrix<STATE_DIM, STATE_DIM> jacobian = {};

        // Position derivatives
        jacobian(0, 0) = 1.0F;
        jacobian(0, 1) = delta_time;
        jacobian(0, 2) = HALF * delta_time * delta_time;

        // --- LOGIC FOR DERIVATIVES (SHARED BETWEEN VELOCITY AND ACCELERATION) ---
        const bool is_potentially_stuck =
          (fabsf(current_velocity) < STUCK_VELOCITY_THRESHOLD) && (fabsf(signed_pwm) > STATIC_FRICTION_THRESHOLD);

        const bool direction_mismatch = (current_velocity * direction_command) < 0.0F;
        float dAcc_dVel = 0.0F;  // Derivative of acceleration w.r.t velocity

        if (fabsf(signed_pwm) > STATIC_FRICTION_THRESHOLD && !is_potentially_stuck)
        {
            if (direction_mismatch && fabsf(current_velocity) > VELOCITY_THRESHOLD)
                dAcc_dVel = -BRAKING_COEFFICIENT;  // Strong braking: a = -v * BRAKING_COEFFICIENT
            else
                dAcc_dVel = -FRICTION_COEFFICIENT;  // Normal friction: a = ... - v * FRICTION_COEFFICIENT
        }
        else
        {
            if (is_potentially_stuck)
                dAcc_dVel =
                  -STATIC_FRICTION_COEFFICIENT;  // Strong static friction: a = -v * STATIC_FRICTION_COEFFICIENT
            else
                dAcc_dVel = -(FRICTION_COEFFICIENT + 1.0F);  // Enhanced braking: a = -v * (FRICTION_COEFFICIENT + 1.0F)
        }

        // Velocity derivatives (dVel_k / dState_k-1)
        jacobian(1, 0) = 0.0F;
        // v_k = v_k-1 + dt * a_k-1(v_k-1, ...)
        // dVel_k / dVel_k-1 = 1 + dt * (dAcc_dVel)
        jacobian(1, 1) = 1.0F + delta_time * dAcc_dVel;
        jacobian(1, 2) = 0.0F;  // Model assumes acceleration doesn't depend on last acceleration

        // Acceleration derivatives (dAcc_k / dState_k-1)
        jacobian(2, 0) = 0.0F;
        // a_k = a_k-1(v_k-1, ...)
        // dAcc_k / dVel_k-1 = dAcc_dVel
        jacobian(2, 1) = dAcc_dVel;
        jacobian(2, 2) = 0.0F;  // Model assumes acceleration doesn't depend on last acceleration

        return jacobian;
    }
    /**
     * @brief Measurement function
     */
    static MeasurementVector measurement_function(const StateVector & state)
    {
        MeasurementVector result = {};
        result[0] = state[0];  // We only measure angle
        return result;
    }

    /**
     * @brief Jacobian of measurement function
     */
    static vt::numeric_matrix<MEAS_DIM, STATE_DIM> measurement_jacobian(const StateVector & /*state*/)
    {
        vt::numeric_matrix<MEAS_DIM, STATE_DIM> jacobian = {};
        jacobian(0, 0) = 1.0F;  // ∂measurement/∂position
        // jacobian(0, 1) = 0.0F;  // ∂measurement/∂velocity
        // jacobian(0, 2) = 0.0F;  // ∂measurement/∂acceleration
        return jacobian;
    }

    /**
     * @brief Normalizes angle while maintaining continuity across wrap-around
     */
    float normalize_angle_with_continuity(float angle_rad)
    {
        const float normalized = normalize_angle(angle_rad);

        // Initialize continuity state on first call
        if (!angle_continuity_initialized_)
        {
            previous_normalized_angle_ = normalized;
            wrapped_angle_offset_ = 0.0F;
            angle_continuity_initialized_ = true;
            return normalized;
        }

        float angle_difference = normalized - previous_normalized_angle_;

        // Handle wrap-around: if jump > PI we've wrapped the other way
        if (angle_difference > PI)
            wrapped_angle_offset_ -= TWO_PI;
        else if (angle_difference < -PI)
            wrapped_angle_offset_ += TWO_PI;

        previous_normalized_angle_ = normalized;
        return normalized + wrapped_angle_offset_;
    }

    /**
     * @brief Updates movement detection state
     */
    bool update_movement_detection(uint32_t current_encoder_count, uint32_t current_time)
    {
        const bool is_moving = (current_encoder_count != last_encoder_count_);

        if (is_moving)
        {
            last_movement_time_ = current_time;
            was_moving_ = true;
        }

        last_encoder_count_ = current_encoder_count;
        return is_moving;
    }

    /**
     * @brief Checks if wheel is stuck (not moving despite PWM command)
     */
    bool is_stuck(uint32_t current_time) const
    {
        if (!was_moving_)
            return false;  // Never moved, not considered stuck

        const uint32_t time_since_movement = (current_time - last_movement_time_) / MILLISECONDS_PER_SECOND;
        return time_since_movement > STUCK_THRESHOLD_MS;
    }

    /**
     * @brief Updates Kalman filter with new measurement
     */
    void update_filter(float measured_angle, uint32_t current_time, int pwm_value, int direction_value)
    {
        float delta_time =
          static_cast<float>(current_time - last_update_us_) / static_cast<float>(MICROSECONDS_PER_SECOND);

        // Ensure reasonable time delta
        if (delta_time > MAX_DELTA_TIME)
            delta_time = INITIAL_DELTA_TIME;

        else if (delta_time < MIN_DELTA_TIME)
            delta_time = MIN_DELTA_TIME;

        if (!initialized_)
        {
            // Initialize filter with proper continuous angle
            state_vector_[0] = normalize_angle_with_continuity(measured_angle);
            state_vector_[1] = 0.0F;
            state_vector_[2] = 0.0F;
            last_update_us_ = current_time;
            initialized_ = true;
        }
        else
        {
            // Predict step
            ControlVector control_input;
            control_input[0] = delta_time;
            control_input[1] = static_cast<float>(pwm_value);
            control_input[2] = static_cast<float>(direction_value);

            filter_.predict(control_input);

            // Update step with normalized angle
            const float continuous_angle = normalize_angle_with_continuity(measured_angle);
            const float predicted_angle = filter_.state_vector[0];

            // Calculate innovation using normalized difference
            const float angle_diff = continuous_angle - predicted_angle;
            const float innovation = normalize_angle(angle_diff);
            const float corrected_measurement = predicted_angle + innovation;

            MeasurementVector measurement;
            measurement[0] = corrected_measurement;
            filter_.update(measurement);

            last_update_us_ = current_time;
        }
    }
};

// Initialize static noise matrices
const vt::numeric_matrix<STATE_DIM, STATE_DIM> WheelKalmanState::process_noise_covariance_ =
  vt::numeric_matrix<STATE_DIM, STATE_DIM>::diagonals({0.001F, 0.1F, 1.0F});

const vt::numeric_matrix<MEAS_DIM, MEAS_DIM> WheelKalmanState::measurement_noise_covariance_ =
  vt::numeric_matrix<MEAS_DIM, MEAS_DIM>::diagonals(0.05F);

// Global Kalman filter instances
WheelKalmanState left_kalman;
WheelKalmanState right_kalman;

// ============================================================================
// HARDWARE ABSTRACTION LAYER
// ============================================================================

/**
 * @brief Direct pin mode configuration for performance
 */
void pinMode_direct(uint8_t pin, uint8_t mode)
{
    if (pin <= MAX_PIN_GROUP_1)
    {
        const uint8_t bit_mask = (1U << (pin & PIN_MASK));
        if (mode == OUTPUT)
            DDRD |= bit_mask;
        else
            DDRD &= ~bit_mask;
    }
    else if (pin <= MAX_PIN_GROUP_2)
    {
        const uint8_t bit_mask = (1U << (pin & PIN_MASK));
        if (mode == OUTPUT)
            DDRB |= bit_mask;
        else
            DDRB &= ~bit_mask;
    }
    else if (pin <= MAX_PIN_GROUP_3)
    {
        const uint8_t bit_mask = (1U << (pin & PIN_MASK));
        if (mode == OUTPUT)
            DDRC |= bit_mask;
        else
            DDRC &= ~bit_mask;
    }
}

/**
 * @brief Direct digital write for performance
 */
void digitalWrite_direct(uint8_t pin, uint8_t value)
{
    if (pin <= MAX_PIN_GROUP_1)
    {
        const uint8_t bit_mask = (1U << (pin & PIN_MASK));
        if (value != 0U)
            PORTD |= bit_mask;
        else
            PORTD &= ~bit_mask;
    }
    else if (pin <= MAX_PIN_GROUP_2)
    {
        const uint8_t bit_mask = (1U << (pin & PIN_MASK));
        if (value != 0U)
            PORTB |= bit_mask;
        else
            PORTB &= ~bit_mask;
    }
    else if (pin <= MAX_PIN_GROUP_3)
    {
        const uint8_t bit_mask = (1U << (pin & PIN_MASK));
        if (value != 0U)
            PORTC |= bit_mask;
        else
            PORTC &= ~bit_mask;
    }
}

/**
 * @brief Analog write for PWM control
 */
void analogWrite_direct(uint8_t pin, uint8_t value)
{
    // Configure Timer0 only once
    static bool pwm_timer0_configured = false;
    if (!pwm_timer0_configured)
    {
        // Configure Timer0 for fast PWM mode.
        TCCR0A = (1U << WGM00) | (1U << WGM01) | (1U << COM0A1) | (1U << COM0B1);
        TCCR0B = (1U << CS01);  // prescaler 8
        pwm_timer0_configured = true;
    }

    switch (pin)
    {
        case 5U:
            OCR0B = value;
            DDRD |= (1U << PD5);
            break;
        case 6U:
            OCR0A = value;
            DDRD |= (1U << PD6);
            break;
        default:
            break;
    }
}

// ===========================================================================
// ADC SETUP FOR ENCODER READINGS
// ===========================================================================

/**
 * @brief Configures ADC for free-running mode on both encoder channels, it will alternate between them
 * and trigger an interrupt on each conversion completion. Therefore, it is non-blocking and the processor
 * can perform other tasks while the ADC is converting.
 */
void setup_adc_free_running()
{
    // REFS0 = 1: Use AVcc as reference, ADLAR = 1: Left adjust result (8-bit mode)
    ADMUX = (1U << REFS0) | (1U << ADLAR) | (current_channel & PIN_MASK);
    // Enable ADC, Auto Trigger Enable, ADC Interrupt, and set prescaler to 64 (faster conversion)
    ADCSRA = (1U << ADEN) | (1U << ADATE) | (1U << ADIE) | (1U << ADPS2) | (1U << ADPS1);
    // Set trigger source to Free Running mode
    ADCSRB = 0U;
    // Start first conversion
    ADCSRA |= (1U << ADSC);
}

/**
 * @brief ADC conversion complete interrupt handler
 *
 * Alternates between left and right encoder channels in round-robin fashion
 */
ISR(ADC_vect)
{
    const uint8_t result = ADCH;

    if (current_channel == LEFT_ENC_PIN)
    {
        left_raw_value = result;
        new_left_data = true;
        current_channel = RIGHT_ENC_PIN;
    }
    else
    {
        right_raw_value = result;
        new_right_data = true;
        current_channel = LEFT_ENC_PIN;
    }

    // Switch to next channel
    ADMUX = (1U << REFS0) | (1U << ADLAR) | (current_channel & PIN_MASK);
}

// ============================================================================
// THRESHOLD MANAGEMENT
// ============================================================================

/**
 * @brief Updates dynamic thresholds for count triggering based on min/max calibration
 */
void update_dynamic_thresholds()
{
    if (left_sample_count >= MIN_SAMPLE_COUNT)
        left_threshold = static_cast<uint8_t>((static_cast<uint16_t>(left_min) + static_cast<uint16_t>(left_max)) / 2U);

    if (right_sample_count >= MIN_SAMPLE_COUNT)
        right_threshold =
          static_cast<uint8_t>((static_cast<uint16_t>(right_min) + static_cast<uint16_t>(right_max)) / 2U);
}

// ============================================================================
// ENCODER PROCESSING AND KALMAN FILTER UPDATES
// ============================================================================

/**
 * @brief Common encoder processing logic for both wheels
 */
void process_encoder_data(volatile uint8_t & raw_value, uint8_t & min_val, uint8_t & max_val, uint8_t & sample_count,
                          uint8_t & threshold, uint8_t & prev_state, int32_t & enc_count, int enc_direction,
                          WheelKalmanState & kalman, int pwm_value)
{
    // Update calibration data
    if (raw_value < min_val)
        min_val = raw_value;

    if (raw_value > max_val)
        max_val = raw_value;

    ++sample_count;

    // Edge detection for incremental counting
    const int new_state = (raw_value >= threshold) ? 1 : 0;

    if (new_state != prev_state)
    {
        enc_count += enc_direction;
        prev_state = new_state;
    }

    // Calculate absolute angle from encoder count
    const float measured_angle = static_cast<float>(enc_count) * RADIANS_PER_COUNT;

    // Update movement detection
    const uint32_t now = micros();
    kalman.update_movement_detection(enc_count, now);

    // Update Kalman filter
    kalman.update_filter(measured_angle, now, pwm_value, enc_direction);
}

/**
 * @brief Processes encoder data and updates Kalman filters for both wheels
 */
void update_encoder_counts_and_speed()
{
    // Process left encoder daWta
    if (new_left_data)
    {
        process_encoder_data(left_raw_value, left_min, left_max, left_sample_count, left_threshold, left_prev_state,
                             left_enc_count, left_direction, left_kalman, left_pwm);
        new_left_data = false;
    }

    // Process right encoder data
    if (new_right_data)
    {
        process_encoder_data(right_raw_value, right_min, right_max, right_sample_count, right_threshold,
                             right_prev_state, right_enc_count, right_direction, right_kalman, right_pwm);
        new_right_data = false;
    }

    update_dynamic_thresholds();
}

// ============================================================================
// SERIAL COMMUNICATION
// ============================================================================

/**
 * @brief Reads and buffers serial data
 */
void readSerialData()
{
    while (Serial.available() > 0 && !newCommand)
    {
        const char character = static_cast<char>(Serial.read());

        if (character == '\n' || character == '\r')
        {
            if (serialIndex > 0U)
            {
                serialBuffer[serialIndex] = '\0';
                newCommand = true;
                serialIndex = 0U;
            }
        }
        else if (serialIndex < (SERIAL_BUFFER_SIZE - 1U))
            serialBuffer[serialIndex++] = character;
        else
            serialIndex = 0U;  // Buffer overflow, reset
    }
}

/**
 * @brief Processes serial commands for motor control
 */
void processSerialCommand()
{
    if (!newCommand)
        return;

    newCommand = false;
    char * pointer = serialBuffer.data();

    // Parse left PWM
    if (*pointer == 'L')
    {
        ++pointer;
        // Use strtol instead of atoi for better error handling
        left_pwm = static_cast<uint8_t>(strtol(pointer, &pointer, 10));

        while (*pointer && *pointer != 'R')
            ++pointer;
    }

    // Parse right PWM
    if (*pointer == 'R')
    {
        ++pointer;
        right_pwm = static_cast<uint8_t>(strtol(pointer, &pointer, 10));

        while (*pointer && *pointer != 'D')
            ++pointer;
    }

    // Parse directions
    if (*pointer == 'D')
    {
        ++pointer;

        // Left direction (LOW = forward, HIGH = backward)
        if (*pointer == '1')
        {
            left_direction = 1;
            digitalWrite_direct(LEFT_DIR_PIN, LOW);
        }
        else if (*pointer == '0')
        {
            left_direction = -1;
            digitalWrite_direct(LEFT_DIR_PIN, HIGH);
        }

        ++pointer;

        // Right direction (HIGH = forward, LOW = backward)
        if (*pointer == '1')
        {
            right_direction = 1;
            digitalWrite_direct(RIGHT_DIR_PIN, HIGH);
        }
        else if (*pointer == '0')
        {
            right_direction = -1;
            digitalWrite_direct(RIGHT_DIR_PIN, LOW);
        }
    }

    // Constrain PWM values and apply to motors
    left_pwm = constrain(left_pwm, 0, 255);
    right_pwm = constrain(right_pwm, 0, 255);

    analogWrite_direct(LEFT_PWM_PIN, left_pwm);
    analogWrite_direct(RIGHT_PWM_PIN, right_pwm);
}

// ============================================================================
// TELEMETRY OUTPUT
// ============================================================================

/**
 * @brief Prints telemetry data for Teleplot visualization and debugging
 */
void print_telemetry_data()
{
    const uint32_t current_time = micros();

    const StateVector & left_state = left_kalman.get_state_vector();
    const StateVector & right_state = right_kalman.get_state_vector();

    // left wheel data
    const float left_encoder_rad = static_cast<float>(left_enc_count) * RADIANS_PER_COUNT;
    const float left_angle_rad = left_state[0];
    const float left_velocity_rad_s = left_state[1];
    const float left_rpm = (left_velocity_rad_s * 60.0F) / TWO_PI;
    const float left_acceleration_rad_s2 = left_state[2];
    const bool is_left_stuck = left_kalman.is_stuck(current_time);
    const float left_pwm_norm = static_cast<float>(left_pwm) / 255.0f;
    const int left_enc_state = left_prev_state ? left_max : left_min;

    // right wheel data
    const float right_encoder_rad = static_cast<float>(right_enc_count) * RADIANS_PER_COUNT;
    const float right_angle_rad = right_state[0];
    const float right_velocity_rad_s = right_state[1];
    const float right_rpm = (right_velocity_rad_s * 60.0F) / TWO_PI;
    const float right_acceleration_rad_s2 = right_state[2];
    const bool is_right_stuck = right_kalman.is_stuck(current_time);
    const float right_pwm_norm = static_cast<float>(right_pwm) / 255.0f;
    const int right_enc_state = right_prev_state ? right_max : right_min;

    // Position data
    Serial.print(">LeftAngleRad:");
    Serial.println(left_angle_rad, 4);
    Serial.print(">RightAngleRad:");
    Serial.println(right_angle_rad, 4);
    Serial.print(">LeftCount:");
    Serial.println(left_enc_count);
    Serial.print(">RightCount:");
    Serial.println(right_enc_count);

    // Speed data
    Serial.print(">LeftVelocityRadSec:");
    Serial.println(left_velocity_rad_s, 4);
    Serial.print(">RightVelocityRadSec:");
    Serial.println(right_velocity_rad_s, 4);
    Serial.print(">LeftRPM:");
    Serial.println(left_rpm, 2);
    Serial.print(">RightRPM:");
    Serial.println(right_rpm, 2);

    // Acceleration data
    Serial.print(">LeftAccelRadSec2:");
    Serial.println(left_acceleration_rad_s2, 4);
    Serial.print(">RightAccelRadSec2:");
    Serial.println(right_acceleration_rad_s2, 4);

    // Motor control data
    Serial.print(">LeftPWMNorm:");
    Serial.println(left_pwm_norm, 3);
    Serial.print(">RightPWMNorm:");
    Serial.println(right_pwm_norm, 3);
    Serial.print(">LeftDir:");
    Serial.println(left_direction);
    Serial.print(">RightDir:");
    Serial.println(right_direction);

    // Movement detection
    Serial.print(">LeftStuck:");
    Serial.println(is_left_stuck);
    Serial.print(">RightStuck:");
    Serial.println(is_right_stuck);

    // Encoder data
    Serial.print(">LeftRaw:");
    Serial.println(left_raw_value);
    Serial.print(">RightRaw:");
    Serial.println(right_raw_value);
    Serial.print(">LeftMin:");
    Serial.println(left_min);
    Serial.print(">LeftMax:");
    Serial.println(left_max);
    Serial.print(">RightMin:");
    Serial.println(right_min);
    Serial.print(">RightMax:");
    Serial.println(right_max);
    Serial.print(">LeftThresh:");
    Serial.println(left_threshold);
    Serial.print(">RightThresh:");
    Serial.println(right_threshold);
    Serial.print(">LeftEncoderRad:");
    Serial.println(left_encoder_rad, 4);
    Serial.print(">RightEncoderRad:");
    Serial.println(right_encoder_rad, 4);
    Serial.print(">LeftEncState:");
    Serial.println(left_enc_state);
    Serial.print(">RightEncState:");
    Serial.println(right_enc_state);
}

// ============================================================================
// ARDUINO MAIN FUNCTIONS
// ============================================================================

void setup()
{
    // Initialize motor control pins
    for (int pin = 5; pin <= 8; ++pin)
        pinMode_direct(pin, OUTPUT);

    // Initialize serial communication
    Serial.begin(BAUD_RATE);

    Serial.println("Teleplot: arduino telemetry");
    Serial.println("Send commands: L<pwm>R<pwm>D<left_dir><right_dir>");
    Serial.println("Example: L128R255D11");

    // Start ADC free-running mode
    setup_adc_free_running();

    // Allow time for ADC stabilization
    delay(INITIAL_DELAY_MS);
}

void loop()
{
    static uint32_t last_telemetry_time = 0U;

    // Main processing pipeline
    readSerialData();
    processSerialCommand();
    update_encoder_counts_and_speed();

    // Telemetry output at 20Hz using non-blocking delay
    if (nonBlockingDelay(last_telemetry_time, TELEMETRY_INTERVAL_MS))
        print_telemetry_data();

    delayMicroseconds(LOOP_DELAY_MICROS);
}
