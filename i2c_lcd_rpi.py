from RPLCD.i2c import CharLCD
import time

lcd = CharLCD(i2c_expander='PCF8574', address=0x27, port=1, cols=16, rows=2, dotsize=8)
lcd.clear()
while(1):
    lcd.clear()
    lcd.write_string('   Welcome')
    lcd.crlf()
    lcd.write_string(' Rahul')
    time.sleep(1)
    