from RPLCD.i2c import CharLCD
from machine import I2C
import time

lcd = CharLCD(i2c_expander='PCF8574', address=0x27, port=1, cols=16, rows=2)
# Clear the display
lcd.clear()

# Write some text
lcd.home()
lcd.write("Hello, world!")
lcd.move(0, 1) # Move to second line
lcd.write("I2C LCD")

# Optional: Turn on backlight
lcd.backlight_on()
time.sleep(3)
lcd.backlight_off()

# Optional: Blink cursor
lcd.cursor_on()
time.sleep(1)
lcd.cursor_off()

# Optional:  Blink the display
lcd.display_on()
time.sleep(1)
lcd.display_off()
