#include <ArduinoSTL.h>
#include <avr/io.h>
#include <util/atomic.h>

#include <array>
#include <cmath>

#undef min
#undef max

#include <FastPID.h>

#include <vt_kalman>
#include <vt_linalg>

// ============================================================================
// CONSTANTS AND CONFIGURATION
// ============================================================================

// --- AVR Pin Definitions (for clarity) ---
// Pin 5 = PD5 (OC0B)
// Pin 6 = PD6 (OC0A)
// Pin 7 = PD7
// Pin 8 = PB0
constexpr uint8_t LEFT_PWM_PIN_REG = 6;   // Mapped to OCR0A
constexpr uint8_t LEFT_DIR_PIN_REG = 8;   // Mapped to PB0
constexpr uint8_t RIGHT_PWM_PIN_REG = 5;  // Mapped to OCR0B
constexpr uint8_t RIGHT_DIR_PIN_REG = 7;  // Mapped to PD7

// Encoder has 8 white sections and 8 cutouts, producing 16 transitions per revolution
constexpr int ENCODER_COUNTS_PER_REVOLUTION = 16;
constexpr float RADIANS_PER_COUNT = TWO_PI / static_cast<float>(ENCODER_COUNTS_PER_REVOLUTION);
constexpr float RAD_S_TO_RPM = 60.0F / (TWO_PI);

// Kalman filter dimensions
constexpr int STATE_DIM = 3;    // [angle, velocity, acceleration]
constexpr int MEAS_DIM = 1;     // Only angle measurement
constexpr int CONTROL_DIM = 3;  // [dt, pwm, direction]

// Serial communication
constexpr int SERIAL_BUFFER_SIZE = 32;
constexpr int SERIAL_TIMEOUT_MS = 10;

// Motor interface pins
constexpr int LEFT_ENC_PIN = 0;
constexpr int RIGHT_ENC_PIN = 1;

// ADC constants
constexpr uint16_t ADC_MAX_VALUE = UINT16_MAX;
constexpr uint16_t ADC_INITIAL_THRESHOLD = ADC_MAX_VALUE / 2;
constexpr uint8_t PIN_MASK = 0x07U;

// Motor model constants
constexpr float PWM_TO_ACCELERATION_GAIN = 0.01F;
constexpr float MAX_ACCELERATION = 200.0F;
constexpr float MIN_ACCELERATION = -200.0F;

constexpr float BRAKING_COEFFICIENT = 5.0F;
constexpr float FRICTION_COEFFICIENT = 0.003F;
constexpr float STATIC_FRICTION_THRESHOLD = 175.0F;
constexpr float STUCK_VELOCITY_THRESHOLD = 0.05F;
constexpr float STUCK_DAMPING_COEFFICIENT = 10.0F;
constexpr float VELOCITY_THRESHOLD = 0.5F;
constexpr float ZERO_RPM_THRESHOLD = VELOCITY_THRESHOLD * RAD_S_TO_RPM;

// Timing constants
#if TELEMETRY
constexpr float MAX_DT = 0.014F;
#else
constexpr float MAX_DT = 0.0100F;
#endif
constexpr float MIN_DT = 0.001F;
constexpr uint32_t INITIAL_DELAY_MS = 100U;
constexpr uint32_t FREQ_CALC_INTERVAL_MS = 1000U;
constexpr uint32_t TELEMETRY_INTERVAL_MS = 200U;  // 5Hz
constexpr uint32_t STUCK_THRESHOLD_MS = 5000U;    // 5 seconds
constexpr uint32_t MICROSECONDS_PER_SECOND = 1000000UL;
constexpr uint32_t MILLISECONDS_PER_SECOND = 1000UL;
constexpr uint32_t LOOP_DELAY_US = 100UL;  // 0.1 ms
// PID control constants
#if TELEMETRY
constexpr uint32_t PID_UPDATE_RATE_HZ = 10U;  // 10 Hz PID update
constexpr uint32_t PID_UPDATE_INTERVAL_MS = 1000U / PID_UPDATE_RATE_HZ;
#else
constexpr uint32_t PID_UPDATE_RATE_HZ = 20U;  // 20 Hz PID update
constexpr uint32_t PID_UPDATE_INTERVAL_MS = 1000U / PID_UPDATE_RATE_HZ;
#endif

constexpr float MIN_TARGET_RPM = 5.0F;
constexpr float MAX_TARGET_RPM = 50.0F;

// PWM slew limiter
constexpr float PWM_SLEW_RATE_PER_MS = 10.0f;  // max pwm change per millisecond
constexpr float BOOST_MULTIPLIER = 1.1F;       // 1.1x the static friction threshold
constexpr uint32_t BOOST_DURATION_US = 150UL;  // 0.15 ms

// Additional constants
constexpr float HALF = 0.5F;

// ============================================================================
// ALIASES FOR EKF
// ============================================================================

using EKF = vt::extended_kalman_filter_t<STATE_DIM, MEAS_DIM, CONTROL_DIM>;
using StateVector = vt::numeric_vector<STATE_DIM>;
using MeasurementVector = vt::numeric_vector<MEAS_DIM>;
using ControlVector = vt::numeric_vector<CONTROL_DIM>;

// ============================================================================
// GLOBAL VARIABLES
// ============================================================================

// --- Encoder ---
volatile uint8_t current_channel = 0U;   // ADC channel being read
volatile uint16_t left_raw_value = 0U;   // Latest ADC value for left encoder
volatile uint16_t right_raw_value = 0U;  // Latest ADC value for right encoder
volatile bool new_left_data = false;     // Flag for new left encoder data
volatile bool new_right_data = false;    // Flag for new right encoder data

// Tracking
int32_t left_enc_count = 0U;
int32_t right_enc_count = 0U;
uint8_t left_prev_state = 0U;
uint8_t right_prev_state = 0U;

// Dynamic threshold calibration
uint16_t left_min = ADC_MAX_VALUE;
uint16_t left_max = 0U;
uint16_t right_min = ADC_MAX_VALUE;
uint16_t right_max = 0U;
uint16_t left_threshold = ADC_INITIAL_THRESHOLD;
uint16_t right_threshold = ADC_INITIAL_THRESHOLD;
uint8_t left_sample_count = 0U;
uint8_t right_sample_count = 0U;

// --- Motor control ---
uint8_t left_pwm = 0U;
uint8_t right_pwm = 0U;
int8_t left_direction = 1;   // 1 for forward, -1 for backward
int8_t right_direction = 1;  // 1 for forward, -1 for backward

// Serial communication
std::array<char, SERIAL_BUFFER_SIZE> serialBuffer;
uint8_t serialIndex = 0U;
bool newCommand = false;

// PID control
float left_target_rpm = 0.0F;
float right_target_rpm = 0.0F;
bool left_pid_enabled = false;
bool right_pid_enabled = false;

// PID gains
float kp = 3.0F;
float ki = 0.5F;
float kd = 6.0F;

// FastPID instances
FastPID left_fast_pid;
FastPID right_fast_pid;

// PID timing
uint32_t last_pid_update_time = 0U;
uint32_t last_pwm_update_us = 0U;

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
 * @note This implementation correctly handles millis() overflow.
 */
inline bool nonBlockingDelay(uint32_t & last_time, const uint32_t & ms_delay)
{
    const uint32_t current_time = millis();
    if (current_time - last_time >= ms_delay)
    {
        last_time = current_time;
        return true;
    }
    return false;
}

/**
 * @brief Sign function for float values
 */
static inline int signf(float x)
{
    return (x > 0.0f) - (x < 0.0f);
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

    // How much we trust the model. Lower values = more trust.
    const vt::numeric_matrix<STATE_DIM, STATE_DIM> process_noise_covariance_ =
      vt::numeric_matrix<STATE_DIM, STATE_DIM>::diagonals({0.001F, 0.5F, 5.0F});
    // How much we trust the sensor. Lower values = more trust.
    const vt::numeric_matrix<MEAS_DIM, MEAS_DIM> measurement_noise_covariance_ =
      vt::numeric_matrix<MEAS_DIM, MEAS_DIM>::diagonals(10.0F);

   public:
    WheelKalmanState()
    : state_vector_(vt::numeric_vector<STATE_DIM>(0.0F)),
      filter_(state_transition_function, state_transition_jacobian, measurement_function, measurement_jacobian,
              process_noise_covariance_, measurement_noise_covariance_, state_vector_)
    {
    }

    // Getters
    const StateVector & get_state_vector() const { return filter_.state_vector; }
    bool is_initialized() const { return initialized_; }
    uint32_t get_last_update_us() const { return last_update_us_; }

    /**
     * @brief State transition function
     */
    static StateVector state_transition_function(const StateVector & prev_state, const ControlVector & control_input)
    {
        // Unpack control vector
        const float dt = control_input[0];
        const float pwm_command = control_input[1];
        const float direction_command = control_input[2];

        // Unpack previous state vector
        const float current_angle = prev_state[0];
        const float current_velocity = prev_state[1];
        const float current_acceleration = prev_state[2];  // Used for position integration

        const float half_dt_squared = HALF * dt * dt;

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
            modeled_acceleration = -current_velocity * STUCK_DAMPING_COEFFICIENT;
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
        next_state[0] = current_angle + dt * current_velocity + half_dt_squared * current_acceleration;

        // v_k = v_k-1 + a_k*dt (Uses the *new* modeled acceleration)
        next_state[1] = current_velocity + dt * modeled_acceleration;

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
        const float dt = control[0];
        const float pwm_command = control[1];
        const float direction_command = control[2];
        const float current_velocity = state[1];
        const float signed_pwm = pwm_command * direction_command;

        vt::numeric_matrix<STATE_DIM, STATE_DIM> jacobian = {};

        // Position derivatives
        jacobian(0, 0) = 1.0F;
        jacobian(0, 1) = dt;
        jacobian(0, 2) = HALF * dt * dt;

        // Determine operating state for acceleration derivative calculation
        const bool is_potentially_stuck =
          (fabsf(current_velocity) < STUCK_VELOCITY_THRESHOLD) && (fabsf(signed_pwm) > STATIC_FRICTION_THRESHOLD);

        // Is the command fighting the current velocity (braking)?
        const bool direction_mismatch = (current_velocity * direction_command) < 0.0F;
        float dAcc_dVel = 0.0F;  // Derivative of acceleration w.r.t velocity

        if (fabsf(signed_pwm) > STATIC_FRICTION_THRESHOLD && !is_potentially_stuck)
        {
            // --- State 1: Normal Operation (Commanded and Moving) ---
            if (direction_mismatch && fabsf(current_velocity) > VELOCITY_THRESHOLD)
                dAcc_dVel = -BRAKING_COEFFICIENT;  // Strong braking: a = -v * BRAKING_COEFFICIENT
            else
                dAcc_dVel = -FRICTION_COEFFICIENT;  // Normal friction: a = ... - v * FRICTION_COEFFICIENT
        }
        else
        {
            // --- State 2 & 3: Stuck or Coasting ---
            if (is_potentially_stuck)
                dAcc_dVel = -STUCK_DAMPING_COEFFICIENT;  // Strong static friction: a = -v * STUCK_DAMPING_COEFFICIENT
            else
                dAcc_dVel = -(FRICTION_COEFFICIENT + 1.0F);  // Enhanced braking: a = -v * (FRICTION_COEFFICIENT + 1.0F)
        }

        // Velocity derivatives (dVel_k / dState_k-1)
        jacobian(1, 0) = 0.0F;
        // v_k = v_k-1 + dt * a_k-1(v_k-1, ...)
        // dVel_k / dVel_k-1 = 1 + dt * (dAcc_dVel)
        jacobian(1, 1) = 1.0F + dt * dAcc_dVel;
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
        float dt = static_cast<float>(current_time - last_update_us_) / static_cast<float>(MICROSECONDS_PER_SECOND);

        // Ensure reasonable time delta for stability
        dt = constrain(dt, MIN_DT, MAX_DT);

        if (!initialized_)
        {
            // Initialize filter
            state_vector_[0] = normalize_angle_with_continuity(measured_angle);
            state_vector_[1] = 0.0F;
            state_vector_[2] = 0.0F;
            last_update_us_ = current_time;
            initialized_ = true;
            return;
        }

        // Predict
        filter_.predict(ControlVector({dt, static_cast<float>(pwm_value), static_cast<float>(direction_value)}));

        // Update using continuous angle
        const float continuous_angle = normalize_angle_with_continuity(measured_angle);
        const float predicted_angle = filter_.state_vector[0];
        const float angle_diff = continuous_angle - predicted_angle;
        const float innovation = normalize_angle(angle_diff);
        const float corrected_prediction = predicted_angle + innovation;

        filter_.update(MeasurementVector(corrected_prediction));

        last_update_us_ = current_time;
    }
};

// Global Kalman filter instances
WheelKalmanState left_kalman;
WheelKalmanState right_kalman;

// ===========================================================================
// ADC SETUP FOR ENCODER READINGS
// ===========================================================================

/**
 * @brief Configures ADC for free-running mode on both encoder channels
 */
void setup_adc_free_running()
{
    // Configure ADC prescaler for 125 kHz ADC clock (16MHz / 128 = 125kHz)
    // ADPS[2:0] = 111 sets prescaler to 128, the slowest speed for max accuracy.
    ADCSRA =
      (1U << ADEN) | (1U << ADATE) | (1U << ADIE) | (1U << ADPS2) | (1U << ADPS1) | (1U << ADPS0);  // Prescaler 128

    // Set trigger source to Free Running mode
    ADCSRB = 0U;

    // Configure first channel with AVcc reference and right-adjusted result (10-bit mode)
    // ADLAR (Left Adjust) is NOT set, so we get the full 10-bit resolution
    ADMUX = (1U << REFS0) | (current_channel & PIN_MASK);

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
    // Read ADCL first, then ADCH to get the 10-bit result
    const uint8_t low = ADCL;
    const uint8_t high = ADCH;
    const uint16_t result = (high << 8) | low;

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
    // ADLAR should NOT be set, as setup configures for 10-bit right-adjusted mode
    ADMUX = (1U << REFS0) | (current_channel & PIN_MASK);
}

// ============================================================================
// THRESHOLD MANAGEMENT
// ============================================================================

/**
 * @brief Updates dynamic thresholds for count triggering based on min/max calibration
 */
void update_dynamic_thresholds()
{
    // Use 32-bit intermediate to prevent overflow when adding two 10-bit numbers
    left_threshold = static_cast<uint16_t>((static_cast<uint32_t>(left_min) + static_cast<uint32_t>(left_max)) / 2U);
    right_threshold = static_cast<uint16_t>((static_cast<uint32_t>(right_min) + static_cast<uint32_t>(right_max)) / 2U);
}

// ============================================================================
// ENCODER PROCESSING AND KALMAN FILTER UPDATES
// ============================================================================

/**
 * @brief Common encoder processing logic for both wheels
 */
void process_encoder_data(volatile uint16_t & raw_value, uint16_t & min_val, uint16_t & max_val, uint8_t & sample_count,
                          uint16_t & threshold, uint8_t & prev_state, int32_t & enc_count, int enc_direction,
                          WheelKalmanState & kalman, int pwm_value)
{
    // Update calibration data
    ATOMIC_BLOCK(ATOMIC_RESTORESTATE)
    {
        if (raw_value < min_val)
            min_val = raw_value;

        if (raw_value > max_val)
            max_val = raw_value;
    }

    ++sample_count;

    // Edge detection for incremental counting
    const int new_state = (raw_value >= threshold) ? 1 : 0;

    if (new_state != prev_state)
    {
        // Encoder count only changes here
        enc_count += enc_direction;
        prev_state = new_state;
    }

    // Current time (microseconds)
    const uint32_t now = micros();
    const float measured_angle = static_cast<float>(enc_count) * RADIANS_PER_COUNT;
    kalman.update_movement_detection(enc_count, now);
    kalman.update_filter(measured_angle, now, pwm_value, enc_direction);
}

/**
 * @brief Processes encoder data and updates Kalman filters for both wheels
 */
void update_encoder_and_ekf()
{
    // Process left encoder data
    if (new_left_data)
    {
        // Must clear flag *before* processing to avoid race condition
        new_left_data = false;
        process_encoder_data(left_raw_value, left_min, left_max, left_sample_count, left_threshold, left_prev_state,
                             left_enc_count, left_direction, left_kalman, left_pwm);
    }

    // Process right encoder data
    if (new_right_data)
    {
        // Must clear flag *before* processing to avoid race condition
        new_right_data = false;
        process_encoder_data(right_raw_value, right_min, right_max, right_sample_count, right_threshold,
                             right_prev_state, right_enc_count, right_direction, right_kalman, right_pwm);
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
    uint8_t chars_read = 0;
    constexpr uint8_t MAX_CHARS_PER_LOOP = 12;

    while (Serial.available() > 0 && !newCommand && chars_read < MAX_CHARS_PER_LOOP)
    {
        const char character = static_cast<char>(Serial.read());
        chars_read++;

        if (character == '\n' || character == '\r')
        {
            if (serialIndex > 0U)
            {
                serialBuffer[serialIndex] = '\0';
                newCommand = true;
                serialIndex = 0U;
                break;  // Exit early when command complete
            }
        }
        else if (serialIndex < (SERIAL_BUFFER_SIZE - 1U))
            serialBuffer[serialIndex++] = character;
        else
            serialIndex = 0U;  // Buffer overflow, reset
    }
}

/**
 * @brief Processes serial commands for motor control and PID setpoints.
 */
void processSerialCommand()
{
    if (!newCommand)
        return;

    newCommand = false;
    char * pointer = serialBuffer.data();

    // Trim leading whitespace
    while (*pointer == ' ' || *pointer == '\t')
        ++pointer;

    // New target command: T<left_rpm>,<right_rpm>
    if (*pointer == 'T')
    {
        ++pointer;
        // parse left rpm (integer)
        char * endptr = nullptr;
        long lrpm = strtol(pointer, &endptr, 10);
        if (endptr != pointer)
        {
            float target = static_cast<float>(lrpm);
            // Constrain target RPM
            if (fabsf(target) < MIN_TARGET_RPM)
                left_target_rpm = 0.0F;  // If below min, set to 0
            else
                left_target_rpm = constrain(target, -MAX_TARGET_RPM, MAX_TARGET_RPM);

            pointer = endptr;
            // skip commas/spaces
            while (*pointer == ',' || *pointer == ' ' || *pointer == '\t')
                ++pointer;
            // parse right rpm
            long rrpm = strtol(pointer, &endptr, 10);
            if (endptr != pointer)
            {
                float target = static_cast<float>(rrpm);
                // Constrain target RPM
                if (fabsf(target) < MIN_TARGET_RPM)
                    right_target_rpm = 0.0F;  // If below min, set to 0
                else
                    right_target_rpm = constrain(target, -MAX_TARGET_RPM, MAX_TARGET_RPM);
            }
            // enable PID for wheels where a valid target was provided
            left_pid_enabled = true;
            right_pid_enabled = true;

            // reset PID controllers to avoid big jumps
            left_fast_pid.clear();
            right_fast_pid.clear();
            Serial.print("T: ");
            Serial.print(left_target_rpm);
            Serial.print(", ");
            Serial.println(right_target_rpm);
        }
        // done processing T-command
        return;
    }

    // PID gains command: P<kp>,<ki>,<kd>
    if (*pointer == 'P')
    {
        ++pointer;
        // parse Kp
        char * endptr = nullptr;
        float new_kp = static_cast<float>(strtod(pointer, &endptr));
        if (endptr != pointer)
        {
            kp = new_kp;  // Update global
            pointer = endptr;

            // skip commas/spaces
            while (*pointer == ',' || *pointer == ' ' || *pointer == '\t')
                ++pointer;

            // parse Ki
            float new_ki = static_cast<float>(strtod(pointer, &endptr));
            if (endptr != pointer)
            {
                ki = new_ki;  // Update global
                pointer = endptr;

                // skip commas/spaces
                while (*pointer == ',' || *pointer == ' ' || *pointer == '\t')
                    ++pointer;

                // parse Kd
                float new_kd = static_cast<float>(strtod(pointer, &endptr));
                if (endptr != pointer)
                    kd = new_kd;  // Update global

                // Re-configure PIDs with new gains
                left_fast_pid.configure(kp, ki, kd, PID_UPDATE_RATE_HZ, 16, true);
                right_fast_pid.configure(kp, ki, kd, PID_UPDATE_RATE_HZ, 16, true);

                Serial.print("Kp, Ki, Kd: ");
                Serial.print(kp);
                Serial.print(", ");
                Serial.print(ki);
                Serial.print(", ");
                Serial.println(kd);

                // Clear PIDs
                left_fast_pid.clear();
                right_fast_pid.clear();
            }
        }
        return;  // Done processing P command
    }

    // legacy manual PWM command: L<left_pwm>R<right_pwm>D<left_dir><right_dir>
    // Parse left PWM (legacy) - disables left PID
    if (*pointer == 'L')
    {
        ++pointer;
        left_pwm = static_cast<uint8_t>(strtol(pointer, &pointer, 10));
        left_pid_enabled = false;  // manual override disables PID for that wheel

        while (*pointer && *pointer != 'R')
            ++pointer;
    }

    // Parse right PWM (legacy) - disables right PID
    if (*pointer == 'R')
    {
        ++pointer;
        right_pwm = static_cast<uint8_t>(strtol(pointer, &pointer, 10));
        right_pid_enabled = false;  // manual override disables PID for that wheel

        while (*pointer && *pointer != 'D')
            ++pointer;
    }

    // Parse directions (legacy)
    if (*pointer == 'D')
    {
        ++pointer;

        // Left direction (LOW = forward, HIGH = backward)
        if (*pointer == '1')
        {
            left_direction = 1;
            PORTB &= ~(1 << PB0);  // Pin 8 LOW
        }
        else if (*pointer == '0')
        {
            left_direction = -1;
            PORTB |= (1 << PB0);  // Pin 8 HIGH
        }

        ++pointer;

        // Right direction (HIGH = forward, LOW = backward)
        if (*pointer == '1')
        {
            right_direction = 1;
            PORTD |= (1 << PD7);  // Pin 7 HIGH
        }
        else if (*pointer == '0')
        {
            right_direction = -1;
            PORTD &= ~(1 << PD7);  // Pin 7 LOW
        }
    }

    // Apply manual PWM outputs only if PID is not enabled for that wheel
    if (!left_pid_enabled)
        OCR0A = left_pwm;  // Pin 6
    if (!right_pid_enabled)
        OCR0B = right_pwm;  // Pin 5
}

// ============================================================================
// TELEMETRY OUTPUT
// ============================================================================

/**
 * @brief Prints telemetry data for Teleplot visualization and debugging
 */
void serial_print_telemetry()
{
    const uint32_t current_time = micros();

    const StateVector & left_state = left_kalman.get_state_vector();
    const StateVector & right_state = right_kalman.get_state_vector();

    // left wheel data
    const float left_encoder_rad = static_cast<float>(left_enc_count) * RADIANS_PER_COUNT;
    const float left_filter_angle_rad = left_state[0];
    const float left_rpm = left_state[1] * RAD_S_TO_RPM;
    const bool is_left_stuck = left_kalman.is_stuck(current_time);
    const float left_pwm_norm = static_cast<float>(left_pwm) / 255.0f;
    const uint16_t left_enc_state = left_prev_state ? left_max : left_min;

    // right wheel data
    const float right_encoder_rad = static_cast<float>(right_enc_count) * RADIANS_PER_COUNT;
    const float right_filter_angle_rad = right_state[0];
    const float right_rpm = right_state[1] * RAD_S_TO_RPM;
    const bool is_right_stuck = right_kalman.is_stuck(current_time);
    const float right_pwm_norm = static_cast<float>(right_pwm) / 255.0f;
    const uint16_t right_enc_state = right_prev_state ? right_max : right_min;

    // Position data
    Serial.print(">a:");
    Serial.println(left_filter_angle_rad, 4);
    Serial.print(">b:");
    Serial.println(right_filter_angle_rad, 4);
    Serial.print(">c:");
    Serial.println(left_encoder_rad, 2);
    Serial.print(">d:");
    Serial.println(right_encoder_rad, 2);

    // Speed data
    Serial.print(">e:");
    Serial.println(left_rpm, 4);
    Serial.print(">f:");
    Serial.println(right_rpm, 4);

    // Motor control data
    Serial.print(">g:");
    Serial.println(left_pwm_norm, 2);
    Serial.print(">h:");
    Serial.println(right_pwm_norm, 2);
    Serial.print(">i:");
    Serial.println(left_direction);
    Serial.print(">j:");
    Serial.println(right_direction);

    // Movement detection
    Serial.print(">k:");
    Serial.println(is_left_stuck);
    Serial.print(">l:");
    Serial.println(is_right_stuck);

    // Encoder data
    ATOMIC_BLOCK(ATOMIC_RESTORESTATE)
    {
        Serial.print(">m:");
        Serial.println(left_raw_value);
        Serial.print(">n:");
        Serial.println(right_raw_value);
    }
    Serial.print(">o:");
    Serial.println(left_min);
    Serial.print(">p:");
    Serial.println(left_max);
    Serial.print(">q:");
    Serial.println(right_min);
    Serial.print(">r:");
    Serial.println(right_max);
    Serial.print(">s:");
    Serial.println(left_threshold);
    Serial.print(">t:");
    Serial.println(right_threshold);
    Serial.print(">u:");
    Serial.println(left_enc_state);
    Serial.print(">v:");
    Serial.println(right_enc_state);

    Serial.print(">z:");
    Serial.println(left_enc_count);
    Serial.print(">y:");
    Serial.println(right_enc_count);
}

// ============================================================================
// PID CONTROL
// ============================================================================

/**
 * @brief Holds state and logic for a single wheel's PID controller
 */
struct PidUpdater
{
    WheelKalmanState & kalman;
    FastPID & pid;
    uint32_t boost_until_us = 0;     // Internal state for startup boost
    float last_measured_rpm = 0.0f;  // State to detect "stuck" vs "passing through"

    PidUpdater(WheelKalmanState & k, FastPID & p) : kalman(k), pid(p) {}

    /**
     * @brief Computes the required PWM and direction for a wheel
     */
    void compute_output(uint32_t now_us, float dt_pwm, float target_rpm, uint8_t & out_pwm, int8_t & out_direction)
    {
        const StateVector & state = kalman.get_state_vector();
        const float measured_rpm = state[1] * RAD_S_TO_RPM;

        // --- Zero RPM Target Override ---
        // If the target is 0 and we are already stopped (or very close),
        // force PWM to 0 to prevent "buzzing" or "hunting" around zero.
        if (target_rpm == 0.0F && fabsf(measured_rpm) < ZERO_RPM_THRESHOLD)
        {
            out_pwm = 0;
            out_direction = 1;  // Direction doesn't matter when PWM is 0
            pid.clear();        // Clear the PID to prevent integral windup from this state
            return;
        }

        // Convert RPM to PID input (scaled for better resolution)
        const int16_t target_pid = static_cast<int16_t>(target_rpm * 10.0f);
        const int16_t measured_pid = static_cast<int16_t>(measured_rpm * 10.0f);

        // --- Feed-Forward and Anti-Windup Logic ---

        // 1. Calculate Feed-Forward
        const float ff_pwm = static_cast<float>(STATIC_FRICTION_THRESHOLD) * signf(target_rpm);

        // Startup boost logic
        const float stuck_rpm_thresh = 1.0f;
        bool is_stuck = fabsf(measured_rpm) < stuck_rpm_thresh;
        bool was_stuck = fabsf(last_measured_rpm) < stuck_rpm_thresh;

        if (is_stuck && was_stuck && fabsf(target_rpm) > 0.5f)
        {
            // We are genuinely stuck at zero, request a boost
            boost_until_us = now_us + BOOST_DURATION_US;
        }
        else if (!is_stuck)
        {
            // We are moving, so clear any pending boost
            boost_until_us = 0;
        }

        last_measured_rpm = measured_rpm;  // Remember for next cycle

        float ff_total = ff_pwm;
        if (now_us < boost_until_us)
        {
            // Apply boost
            ff_total = signf(target_rpm) * (STATIC_FRICTION_THRESHOLD * BOOST_MULTIPLIER);
        }

        // 2. Set Dynamic Output Range for Anti-Windup
        // The PID output must fit in the remaining room *after* feed-forward is applied.
        // We scale by 10 for the PID's integer math.
        const int16_t ff_scaled = static_cast<int16_t>(ff_total * 10.0f);
        const int16_t max_output_scaled = 2550;
        const int16_t min_output_scaled = -2550;

        // The PID output range is now asymmetric
        pid.setOutputRange(min_output_scaled - ff_scaled, max_output_scaled - ff_scaled);

        // 3. Get PID output (which is now correctly clamped by FastPID's internal anti-windup)
        int16_t pid_output = pid.step(target_pid, measured_pid);

        // 4. Combine PID output and Feed-Forward
        // We use the scaled values for combination *before* clamping
        int32_t u_scaled = static_cast<int32_t>(pid_output) + static_cast<int32_t>(ff_scaled);

        // 5. Clamp the *final* combined output
        if (u_scaled > max_output_scaled)
            u_scaled = max_output_scaled;
        else if (u_scaled < min_output_scaled)
            u_scaled = min_output_scaled;

        // Convert scaled output back to float
        float u = static_cast<float>(u_scaled) / 10.0f;

        // 6. Slew-rate limit the *signed* output
        float max_delta = PWM_SLEW_RATE_PER_MS * dt_pwm;
        // out_pwm is the current magnitude, out_direction is the current direction
        float current_signed_pwm = static_cast<float>(out_pwm) * static_cast<float>(out_direction);
        // u is the desired signed pwm
        float pwm_diff = u - current_signed_pwm;

        if (pwm_diff > max_delta)
            u = current_signed_pwm + max_delta;
        else if (pwm_diff < -max_delta)
            u = current_signed_pwm - max_delta;

        // 7. Set final output variables from the slew-limited signed value
        out_pwm = static_cast<uint8_t>(constrain(static_cast<int>(fabsf(u)), 0, 255));
        out_direction = (u >= 0.0f) ? 1 : -1;
    }
};

// Create global instances of the PID updater struct
PidUpdater left_pid_updater(left_kalman, left_fast_pid);
PidUpdater right_pid_updater(right_kalman, right_fast_pid);

/**
 * @brief Applies PID control outputs to both motors
 * @note This is now much simpler and delegates logic to PidUpdater
 */
void apply_pid_control_once()
{
    const uint32_t now_us = micros();
    float dt_pwm = MAX_DT;  // Default to 1ms
    if (last_pwm_update_us != 0U)
        dt_pwm = (now_us - last_pwm_update_us) / 1e3f;  // ms
    last_pwm_update_us = now_us;

    // LEFT
    if (left_pid_enabled)
    {
        left_pid_updater.compute_output(now_us, dt_pwm, left_target_rpm, left_pwm, left_direction);

        // write direction (LOW = forward, HIGH = backward)
        if (left_direction == 1)
            PORTB &= ~(1 << PB0);  // Pin 8 LOW
        else
            PORTB |= (1 << PB0);  // Pin 8 HIGH

        // write pwm
        OCR0A = left_pwm;  // Pin 6
    }

    // Right wheel
    if (right_pid_enabled)
    {
        right_pid_updater.compute_output(now_us, dt_pwm, right_target_rpm, right_pwm, right_direction);

        // write direction (HIGH = forward, LOW = backward)
        if (right_direction == 1)
            PORTD |= (1 << PD7);  // Pin 7 HIGH
        else
            PORTD &= ~(1 << PD7);  // Pin 7 LOW

        // write pwm
        OCR0B = right_pwm;  // Pin 5
    }
}

// ============================================================================
// ARDUINO MAIN FUNCTIONS
// ============================================================================

void setup()
{
    // Initialize motor control pins as outputs
    // (Pin 5, 6, 7 are on PORTD; Pin 8 is on PORTB)
    DDRD |= (1 << PD5) | (1 << PD6) | (1 << PD7);
    DDRB |= (1 << PB0);

    // Configure Timer0 for fast PWM mode on pins 5 (OC0B) and 6 (OC0A)
    // Prescaler = 8 (approx 7.8 kHz PWM frequency)
    TCCR0A = (1U << WGM00) | (1U << WGM01) | (1U << COM0A1) | (1U << COM0B1);
    TCCR0B = (1U << CS01);

    // Initialize serial communication
    Serial.begin(BAUD_RATE);
    Serial.setTimeout(SERIAL_TIMEOUT_MS);

    Serial.println("Teleplot: arduino telemetry");
    Serial.println("Send commands: L<pwm>R<pwm>D<left_dir><right_dir>");
    Serial.println("Example: L128R255D11 (max: 255), dir: 1=forward,0=backward");
    Serial.println("Set RPM targets: T<left_rpm>,<right_rpm>  (example: T20,-50), min/max: +- 5/50");
    Serial.println("Set PID gains: P<kp>,<ki>,<kd>  (example: P3.0,1.0,6.0)");

    // Configure FastPID controllers
    // Parameters: kp, ki, kd, hz, bits, sign
    left_fast_pid.configure(kp, ki, kd, PID_UPDATE_RATE_HZ, 16, true);
    right_fast_pid.configure(kp, ki, kd, PID_UPDATE_RATE_HZ, 16, true);

    // Set output range for PID (scaled by 10 for better resolution)
    left_fast_pid.setOutputRange(-2550, 2550);  // -255 to 255 scaled by 10
    right_fast_pid.setOutputRange(-2550, 2550);

    // Start ADC free-running mode
    setup_adc_free_running();

    // Allow time for ADC stabilization
    delay(INITIAL_DELAY_MS);
}

void loop()
{
    // Update encoders and EKF
    update_encoder_and_ekf();

    // Service serial
    readSerialData();
    processSerialCommand();

    // Apply PID control
    apply_pid_control_once();

    // Small delay to yield the CPU
    delayMicroseconds(LOOP_DELAY_US);

#if TELEMETRY
    static uint32_t last_telemetry_time = 0U;
    if (nonBlockingDelay(last_telemetry_time, TELEMETRY_INTERVAL_MS))
        serial_print_telemetry();
#endif
}
