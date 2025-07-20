from hd44780.hd44780 import I2CDisplay
from machine import I2C
import time

# Initialize I2C (assuming I2C bus 0 and address 0x27, adjust as needed)
i2c = I2C(0, scl=21, sda=20, freq=400000)  # Example for an ESP32, adjust pins and freq
lcd = I2CDisplay(i2c, cols=16, rows=2, i2c_addr=0x27)  # Adjust i2c_addr if needed

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
