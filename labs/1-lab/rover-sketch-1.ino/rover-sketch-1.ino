/* Copy and paste the code below into the Arduino software */
int M1 = 6; //M1 Speed Control
int D1 = 8; //M1 Direction Control

int M2 = 5; //M2 Speed Control
int D2 = 7; //M2 Direction Control

void setup()
{
    int i;
    for(i=5;i<=8;i++)
    pinMode(i, OUTPUT);
    Serial.begin(9600);
}
void loop()
{
    int leftspeed = 255; //255 is maximum speed
    int rightspeed = 255;

    analogWrite (M1,255);
    digitalWrite(D1,HIGH);

    analogWrite (M2,255);
    digitalWrite(D2,LOW);

    delay(100);
}