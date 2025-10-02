/* The code below simply counts the number of changes, so a disc with 8x white sections and 8x cutouts will provide a M1_enc_count of 16 per 360 degree rotation. It is up to you to integrate it with
your code*/
int rawsensorValue = 0; // variable to store the value coming from the sensor

int M1_pwm_pin = 6;
int M1_dir_pin = 8;
int M1_enc_pin = 1; // Analog pin 0
int M1_enc_c0 = 0;
int M1_enc_c1 = 0;
long M1_enc_count = 0;

int M2_pwm_pin = 5;
int M2_dir_pin = 7;
int M2_enc_pin = 0; // Analog pin 1
int M2_enc_c0 = 0;
int M2_enc_c1 = 0;
long M2_enc_count = 0;


void setup() {
    for(int i=5;i<=8;i++)
        pinMode(i, OUTPUT);

    Serial.begin(9600);

    int leftspeed = 255; //255 is maximum speed
    int rightspeed = 255;
}

void loop() {
    analogWrite (M1_pwm_pin,255);
    digitalWrite (M1_dir_pin,LOW);

    analogWrite (M2_pwm_pin,255);
    digitalWrite (M2_dir_pin,HIGH);

    delay(20);

    rawsensorValue = analogRead(M1_enc_pin);

    if (rawsensorValue < 600){
         //Min value is 400 and max value is 800, so state chance can be done at 600.
        M1_enc_c1 = 1;
    }
    else {
        M1_enc_c1 = 0;
    }
    if (M1_enc_c1 != M1_enc_c0){
        M1_enc_count ++;
    }

    M1_enc_c0 = M1_enc_c1;

    rawsensorValue = analogRead(M2_enc_pin);

    if (rawsensorValue < 600){
         //Min value is 400 and max value is 800, so state chance can be done at 600.
        M2_enc_c1 = 1;
    }
    else {
        M2_enc_c1 = 0;
    }
    if (M2_enc_c1 != M2_enc_c0){
        M2_enc_count ++;
    }

    M2_enc_c0 = M2_enc_c1;

    Serial.println("M1 enc count: " + String(M1_enc_count) + " M2 enc count: " + String(M2_enc_count));
}