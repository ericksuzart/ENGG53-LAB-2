#include <Arduino.h>

/* The code below simply counts the number of changes, so a disc with 8x white sections and 8x cutouts will provide a
left_enc_count of 16 per 360 degree rotation. It is up to you to integrate it with your code*/

int rawsensorValue = 0;  // variable to store the value coming from the sensor

int left_pwm_pin = 6;
int left_dir_pin = 8;
int left_enc_pin = 1;  // Analog pin 0
int left_enc_c0 = 0;
int left_enc_c1 = 0;
long left_enc_count = 0;

int right_pwm_pin = 5;
int right_dir_pin = 7;
int right_enc_pin = 0;  // Analog pin 1
int right_enc_c0 = 0;
int right_enc_c1 = 0;
long right_enc_count = 0;

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

uint16_t analogRead_blocking(uint8_t channel)
{
    // Select Vref = AVcc, right adjust
    ADMUX = (1 << REFS0) | (channel & 0x07);

    // Enable ADC and set prescaler to 128 (16MHz /128 = 125kHz)
    ADCSRA = (1 << ADEN) | (1 << ADPS2) | (1 << ADPS1) | (1 << ADPS0);

    // Start conversion
    ADCSRA |= (1 << ADSC);

    // Wait for completion
    while (ADCSRA & (1 << ADSC))
        continue;

    uint16_t result = ADC;  // ADCL + ADCH combined
    return result;
}

void setup()
{
    for (int i = 5; i <= 8; i++)
        pinMode_direct(i, OUTPUT);

    Serial.begin(BAUD_RATE);
}

void loop()
{
    // 255 is maximum speed
    int left_speed = 255;
    int right_speed = 255;

    analogWrite_direct(left_pwm_pin, left_speed);
    digitalWrite_direct(left_dir_pin, LOW);

    analogWrite_direct(right_pwm_pin, right_speed);
    digitalWrite_direct(right_dir_pin, HIGH);

    delay(20);

    rawsensorValue = analogRead_blocking(left_enc_pin);

    if (rawsensorValue < 600)
        // Min value is 400 and max value is 800, so state change can be done at 600.
        left_enc_c1 = 1;
    else
        left_enc_c1 = 0;

    if (left_enc_c1 != left_enc_c0)
        left_enc_count++;

    left_enc_c0 = left_enc_c1;

    rawsensorValue = analogRead_blocking(right_enc_pin);

    // Min value is 400 and max value is 800, so state change can be done at 600.
    if (rawsensorValue < 600)
        right_enc_c1 = 1;
    else
        right_enc_c1 = 0;

    if (right_enc_c1 != right_enc_c0)
        right_enc_count++;

    right_enc_c0 = right_enc_c1;

    Serial.println("Left enc count: " + String(left_enc_count) + " Right enc count: " + String(right_enc_count));
}
