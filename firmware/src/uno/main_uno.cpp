#include <Arduino.h>

/* The encoder has 8 white sections and 8 cutouts, producing 16 transitions per revolution */

// Variables to store the values coming from the sensors
volatile uint16_t left_raw_value = 0;
volatile uint16_t right_raw_value = 0;
volatile bool new_left_data = false;
volatile bool new_right_data = false;

// Moving average filter parameters for ADC
#define FILTER_WINDOW_SIZE 8  // Power of 2 for efficient division

// Moving average buffers and indices for ADC
uint16_t left_filter_buffer[FILTER_WINDOW_SIZE] = {0};
uint16_t right_filter_buffer[FILTER_WINDOW_SIZE] = {0};

uint8_t left_filter_index = 0;
uint8_t right_filter_index = 0;

uint32_t left_filter_sum = 0;
uint32_t right_filter_sum = 0;

uint16_t left_filtered_value = 0;
uint16_t right_filtered_value = 0;

// Dynamic threshold variables
uint16_t left_min = 1023;  // Start with maximum possible value
uint16_t left_max = 0;     // Start with minimum possible value
uint16_t right_min = 1023;
uint16_t right_max = 0;
uint16_t left_threshold = 512;   // Initial midpoint
uint16_t right_threshold = 512;  // Initial midpoint

// Wheel parameters
constexpr int ENCODER_COUNTS_PER_REVOLUTION = 16;  // 8 white + 8 black sections
constexpr float RADIANS_PER_COUNT = 2.0 * PI / ENCODER_COUNTS_PER_REVOLUTION;

// Speed calculation variables - EDGE TIMING APPROACH
unsigned long left_last_edge_time = 0;
unsigned long right_last_edge_time = 0;
unsigned long left_current_edge_time = 0;
unsigned long right_current_edge_time = 0;

float left_angular_speed = 0.0;   // radians per second
float right_angular_speed = 0.0;  // radians per second
float left_rpm = 0.0;             // revolutions per minute
float right_rpm = 0.0;            // revolutions per minute

int left_direction = 1;   // from motor driver (1 = forward, -1 for backward)
int right_direction = 1;  // from motor driver (1 = forward, -1 for backward)

// Minimum time between edges to avoid noise (microseconds)
constexpr unsigned long MIN_EDGE_TIME_US = 1000;  // 1ms minimum

// Threshold adaptation parameters
constexpr uint16_t MIN_SAMPLE_COUNT = 50;  // Number of samples before adapting threshold
uint16_t left_sample_count = 0;
uint16_t right_sample_count = 0;

// Motor control pins
int left_pwm_pin = 6;
int left_dir_pin = 8;
int left_enc_pin = 0;  // Analog pin A0
int left_enc_c0 = 0;
int left_enc_c1 = 0;
long left_enc_count = 0;

int right_pwm_pin = 5;
int right_dir_pin = 7;
int right_enc_pin = 1;  // Analog pin A1
int right_enc_c0 = 0;
int right_enc_c1 = 0;
long right_enc_count = 0;

// Channel tracking for round-robin sampling
volatile uint8_t current_channel = 0;

void pinMode_direct(uint8_t pin, uint8_t mode)
{
    volatile uint8_t * ddr;

    if (pin >= 0 && pin <= 7)
        ddr = &DDRD;
    else if (pin >= 8 && pin <= 13)
        ddr = &DDRB;
    else if (pin >= 14 && pin <= 19)
        ddr = &DDRC;
    else
        return;

    uint8_t bit = (pin % 8);

    if (mode == OUTPUT)
        *ddr |= (1 << bit);
    else
        *ddr &= ~(1 << bit);
}

void digitalWrite_direct(uint8_t pin, uint8_t val)
{
    volatile uint8_t * port;

    if (pin >= 0 && pin <= 7)
        port = &PORTD;
    else if (pin >= 8 && pin <= 13)
        port = &PORTB;
    else if (pin >= 14 && pin <= 19)
        port = &PORTC;
    else
        return;

    uint8_t bit = (pin % 8);

    if (val)
        *port |= (1 << bit);  // Drive HIGH
    else
        *port &= ~(1 << bit);  // Drive LOW
}

void analogWrite_direct(uint8_t pin, uint8_t val)
{
    // Configure Fast PWM, non-inverting
    TCCR0A = (1 << WGM00) | (1 << WGM01) | (1 << COM0A1) | (1 << COM0B1);
    TCCR0B = (1 << CS01);

    switch (pin)
    {
        case 5:
            OCR0B = val;
            DDRD |= (1 << PD5);
            break;
        case 6:
            OCR0A = val;
            DDRD |= (1 << PD6);
            break;
        default:
            break;
    }
}

// Moving average filter function for ADC values only
uint16_t apply_moving_average(uint16_t new_value, uint16_t buffer[], uint8_t * index, uint32_t * sum,
                              uint8_t window_size)
{
    // Subtract the oldest value from the sum
    *sum -= buffer[*index];

    // Add the new value to the sum
    *sum += new_value;

    // Store the new value in the buffer
    buffer[*index] = new_value;

    // Move to the next position in the circular buffer
    *index = (*index + 1) % window_size;

    // Return the average (using bit shift for division by power of 2)
    return *sum >> 3;  // Equivalent to division by 8 (FILTER_WINDOW_SIZE)
}

// Update dynamic thresholds based on min/max values
void update_dynamic_thresholds()
{
    // Update threshold after collecting enough samples and calculate threshold as midpoint between
    // min and max
    if (left_sample_count >= MIN_SAMPLE_COUNT)
        left_threshold = (left_min + left_max) / 2;

    if (right_sample_count >= MIN_SAMPLE_COUNT)
        right_threshold = (right_min + right_max) / 2;
}

// Calculate speed based on time between encoder edges
void calculate_speed_from_edge_timing()
{
    // Calculate left wheel speed when we have valid edge timing
    if (left_last_edge_time > 0 && left_current_edge_time > left_last_edge_time)
    {
        unsigned long period = left_current_edge_time - left_last_edge_time;

        // Only calculate if reasonable time has passed (avoid noise and division by zero)
        if (period >= MIN_EDGE_TIME_US)
        {
            float period_seconds = period / 1000000.0;  // Convert to seconds
            float frequency = 1.0 / period_seconds;     // Frequency in Hz (edges per second)

            // Convert to angular speed (radians per second)
            // Each edge represents 1 count, 16 counts per revolution
            left_angular_speed = (frequency / ENCODER_COUNTS_PER_REVOLUTION) * 2.0 * PI * left_direction;

            // Convert to RPM
            left_rpm = (frequency / ENCODER_COUNTS_PER_REVOLUTION) * 60.0 * left_direction;
        }
    }

    // Calculate right wheel speed when we have valid edge timing
    if (right_last_edge_time > 0 && right_current_edge_time > right_last_edge_time)
    {
        unsigned long period = right_current_edge_time - right_last_edge_time;

        // Only calculate if reasonable time has passed
        if (period >= MIN_EDGE_TIME_US)
        {
            float period_seconds = period / 1000000.0;  // Convert to seconds
            float frequency = 1.0 / period_seconds;     // Frequency in Hz (edges per second)

            // Convert to angular speed (radians per second)
            right_angular_speed = (frequency / ENCODER_COUNTS_PER_REVOLUTION) * 2.0 * PI * right_direction;

            // Convert to RPM
            right_rpm = (frequency / ENCODER_COUNTS_PER_REVOLUTION) * 60.0 * right_direction;
        }
    }
}

void setup_adc_free_running()
{
    // Configure ADC for Free Running Mode
    // REFS0 = 1: Use AVcc as reference, right adjust result (ADLAR = 0)
    ADMUX = (1 << REFS0) | (current_channel & 0x07);  // Start with left channel (A0)

    // Enable ADC, Auto Trigger Enable, ADC Interrupt, and set prescaler to 128 (16MHz/128 = 125kHz)
    ADCSRA = (1 << ADEN) | (1 << ADATE) | (1 << ADIE) | (1 << ADPS2) | (1 << ADPS1) | (1 << ADPS0);

    // Set trigger source to Free Running mode (ADTS2:0 = 0)
    ADCSRB = 0;

    // Start the first conversion to kick things off
    ADCSRA |= (1 << ADSC);
}

ISR(ADC_vect)
{
    // This ISR is called when an ADC conversion is complete
    uint16_t result = ADC;  // Read the result (this also clears ADIF flag)

    // Store result based on current channel
    if (current_channel == left_enc_pin)
    {
        left_raw_value = result;
        new_left_data = true;

        // Switch to right channel for next conversion
        current_channel = right_enc_pin;
    }
    else
    {
        right_raw_value = result;
        new_right_data = true;

        // Switch to left channel for next conversion
        current_channel = left_enc_pin;
    }

    // Update MUX for next conversion
    ADMUX = (1 << REFS0) | (current_channel & 0x07);

    // In free-running mode, the next conversion starts automatically
}

void update_encoder_counts_and_speed()
{
    // Update left encoder if new data is available
    if (new_left_data)
    {
        // Apply moving average filter to left encoder ADC reading
        left_filtered_value = apply_moving_average(left_raw_value, left_filter_buffer, &left_filter_index,
                                                   &left_filter_sum, FILTER_WINDOW_SIZE);

        // Update left min/max values
        if (left_filtered_value < left_min)
            left_min = left_filtered_value;

        if (left_filtered_value > left_max)
            left_max = left_filtered_value;

        left_sample_count++;

        // Update dynamic threshold
        update_dynamic_thresholds();

        // Use dynamic threshold for edge detection
        int new_state = (left_filtered_value < left_threshold) ? 1 : 0;

        // Detect edge (state change from white to black or black to white)
        if (new_state != left_enc_c0)
        {
            // Increment or decrement based on direction for absolute odometry
            left_enc_count += left_direction;

            // Update timing for speed calculation
            left_last_edge_time = left_current_edge_time;
            left_current_edge_time = micros();
        }

        left_enc_c0 = new_state;
        new_left_data = false;
    }

    // Update right encoder if new data is available
    if (new_right_data)
    {
        // Apply moving average filter to right encoder ADC reading
        right_filtered_value = apply_moving_average(right_raw_value, right_filter_buffer, &right_filter_index,
                                                    &right_filter_sum, FILTER_WINDOW_SIZE);

        // Update right min/max values
        if (right_filtered_value < right_min)
            right_min = right_filtered_value;

        if (right_filtered_value > right_max)
            right_max = right_filtered_value;

        right_sample_count++;

        // Update dynamic threshold
        update_dynamic_thresholds();

        // Use dynamic threshold for edge detection
        int new_state = (right_filtered_value < right_threshold) ? 1 : 0;

        // Detect edge (state change from white to black or black to white)
        if (new_state != right_enc_c0)
        {
            // Increment or decrement based on direction for absolute odometry
            right_enc_count += right_direction;

            // Update timing for speed calculation
            right_last_edge_time = right_current_edge_time;
            right_current_edge_time = micros();
        }

        right_enc_c0 = new_state;
        new_right_data = false;
    }

    // Calculate speeds based on edge timing
    calculate_speed_from_edge_timing();
}

void setup()
{
    for (int i = 5; i <= 8; i++)
        pinMode_direct(i, OUTPUT);

    Serial.begin(BAUD_RATE);

    // Initialize moving average buffers for ADC readings
    for (int i = 0; i < FILTER_WINDOW_SIZE; i++)
    {
        left_filter_buffer[i] = 512;   // Midpoint value
        right_filter_buffer[i] = 512;  // Midpoint value
    }

    left_filter_sum = 512 * FILTER_WINDOW_SIZE;
    right_filter_sum = 512 * FILTER_WINDOW_SIZE;

    // Initialize ADC in free-running mode
    setup_adc_free_running();
}

void loop()
{
    // 255 is maximum speed
    int left_speed = 255;
    int right_speed = 255;

    // Set motor directions and update direction variables
    digitalWrite_direct(left_dir_pin, LOW);
    left_direction = 1;  // LOW means forward in this configuration

    digitalWrite_direct(right_dir_pin, HIGH);
    right_direction = -1;  // HIGH means backward in this configuration

    analogWrite_direct(left_pwm_pin, left_speed);
    analogWrite_direct(right_pwm_pin, right_speed);

    // Update encoder counts and calculate speeds
    update_encoder_counts_and_speed();

    // Print data for Arduino Plotter
    static unsigned long last_print = 0;
    if (millis() - last_print > 100)  // Print every 100ms
    {
        Serial.print("LeftRPM:");
        Serial.print(left_rpm, 2);
        Serial.print(",");
        Serial.print("RightRPM:");
        Serial.print(right_rpm, 2);
        Serial.print(",");
        Serial.print("LeftDir:");
        Serial.print(left_direction);
        Serial.print(",");
        Serial.print("RightDir:");
        Serial.print(right_direction);
        Serial.print(",");
        Serial.print("LeftCount:");
        Serial.print(left_enc_count);
        Serial.print(",");
        Serial.print("RightCount:");
        Serial.println(right_enc_count);

        last_print = millis();
    }

    // Small delay to prevent overwhelming the main loop
    delay(1);
}
