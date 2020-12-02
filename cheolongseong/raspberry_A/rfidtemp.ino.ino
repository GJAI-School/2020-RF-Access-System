/*
   -----------------------------------------------------------------------------------------
               MFRC522      Arduino       Arduino   Arduino    Arduino          Arduino
               Reader/PCD   Uno/101       Mega      Nano v3    Leonardo/Micro   Pro Micro
   Signal      Pin          Pin           Pin       Pin        Pin              Pin
   -----------------------------------------------------------------------------------------
   RST/Reset   RST          9             5         D9         RESET/ICSP-5     RST
   SPI SS      SDA(SS)      10            53        D10        10               10
   SPI MOSI    MOSI         11 / ICSP-4   51        D11        ICSP-4           16
   SPI MISO    MISO         12 / ICSP-1   50        D12        ICSP-1           14
   SPI SCK     SCK          13 / ICSP-3   52        D13        ICSP-3           15
*/
#include <deprecated.h>
#include <require_cpp11.h>
#include <MFRC522Extended.h>

#include <SPI.h>
#include <MFRC522.h>
#define SS_PIN 10    // spi 통신을 위한 SS(chip select)핀 설정
#define RST_PIN 9    // 리셋 핀 설정

#include "IRTemp.h"
static const byte PIN_DATA    = 2; // Choose any pins you like for these
static const byte PIN_CLOCK   = 3;
static const byte PIN_ACQUIRE = 4;
static const TempUnit SCALE = CELSIUS; // Options are CELSIUS, FAHRENHEIT
IRTemp irTemp(PIN_ACQUIRE, PIN_CLOCK, PIN_DATA);

MFRC522 rfid(SS_PIN, RST_PIN); // 'rfid' 이름으로 클래스 객체 선언
MFRC522::MIFARE_Key key;

byte nuidPICC[4];   // 카드 ID들을 저장(비교)하기 위한 배열(변수)선언

void setup() {
  Serial.begin(9600);
  SPI.begin(); // SPI 통신 시작
  rfid.PCD_Init(); // RFID(MFRC522) 초기화
  // ID값 초기화
  for (byte i = 0; i < 6; i++) {
    key.keyByte[i] = 0xFF;
  }
  // MIFARE 타입의 카드키 종류들만 인식됨을 표시

  //printHex(key.keyByte, MFRC522::MF_KEY_SIZE);
}

void loop() {
  //Serial.println("None");
  // 새카드 접촉이 있을 때만 다음 단계로 넘어감
  if ( ! rfid.PICC_IsNewCardPresent())
    return;

  // 카드 읽힘이 제대로 되면 다음으로 넘어감
  if ( ! rfid.PICC_ReadCardSerial())
    return;
  // 현재 접촉 되는 카드 타입을 읽어와 모니터에 표시함

  MFRC522::PICC_Type piccType = rfid.PICC_GetType(rfid.uid.sak);
  Serial.print(rfid.PICC_GetTypeName(piccType));
  //rfid.PICC_GetTypeName(piccType);

  // MIFARE 방식의 카드인지 확인 루틴
  if (piccType != MFRC522::PICC_TYPE_MIFARE_MINI &&
      piccType != MFRC522::PICC_TYPE_MIFARE_1K &&
      piccType != MFRC522::PICC_TYPE_MIFARE_4K) {

    return;
  }

  // 이전 인식된 카드와 다른, 혹은 새카드가 인식되면
  /*if (rfid.uid.uidByte[0] != nuidPICC[0] ||
    rfid.uid.uidByte[1] != nuidPICC[1] ||
    rfid.uid.uidByte[2] != nuidPICC[2] ||
    rfid.uid.uidByte[3] != nuidPICC[3] ) {
    Serial.println(F("Resgisted New card"));
  */
  // 고유아이디(UID) 값을 저장한다.
  for (byte i = 0; i < 4; i++) {
    nuidPICC[i] = rfid.uid.uidByte[i];
  }
  // 그 UID 값을 16진값으로 출력 한다.
  //Serial.println(F("The NUID tag is:"));
  //Serial.print(F("In hex: "));
  //printHex(rfid.uid.uidByte, rfid.uid.size);
  //Serial.println();
  // 그 UID 값을 10진값으로 출력 한다.
  //Serial.print(F("In dec: "));
  printDec(rfid.uid.uidByte, rfid.uid.size);
  //Serial.println();
  //}
  // 연속으로 동일한 카드를 접촉하면 다른 처리 없이
  // '이미 인식된 카드'라는 메세지를 출력한다.
  // else Serial.println(F("Hello"));
  rfid.PICC_HaltA();
  rfid.PCD_StopCrypto1();
  float irTemperature = irTemp.getIRTemperature(SCALE);
  //printTemperature("Object", irTemperature + 7);
  float ambientTemperature = irTemp.getAmbientTemperature(SCALE);
  //printTemperature("Atmosphere", ambientTemperature);

  Serial.print("::");
  Serial.println(irTemperature+11);

  delay(500);
}
// 16진 값으로 변환 해주는 함수 정의
void printHex(byte *buffer, byte bufferSize) {
  for (byte i = 0; i < bufferSize; i++) {
    Serial.print(buffer[i] < 0x10 ? " 0" : " ");
    Serial.print(buffer[i], HEX);
  }
}
// 10진 값으로 변환 해주는 함수 정의
void printDec(byte *buffer, byte bufferSize) {
  for (byte i = 0; i < bufferSize; i++) {
    Serial.print(buffer[i] < 0x10 ? " 0" : " ");
    Serial.print(buffer[i], DEC);
  }
}
void printTemperature(
  char  *type,
  float  temperature) {

  Serial.print(type);
  Serial.print("temp: ");

  if (isnan(temperature)) {
    Serial.println("Failed");
  }
  else {
    Serial.print(temperature);
    Serial.println(SCALE == FAHRENHEIT  ?  " F"  :  " C");
  }
}
