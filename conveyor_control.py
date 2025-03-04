import RPi.GPIO as GPIO
import time
import requests

# Pin GPIO untuk sensor
sensor_pin = 17
sensor2_pin = 27

# Pin GPIO untuk kontrol motor (IBT-2)
pwm_pin_1 = 12  # RPWM/PWM1 
pwm_pin_2 = 13

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(sensor_pin, GPIO.IN)
GPIO.setup(sensor2_pin, GPIO.IN)
GPIO.setup(pwm_pin_1, GPIO.OUT)
GPIO.setup(pwm_pin_2, GPIO.OUT)

# Setup PWM
pwm1 = GPIO.PWM(pwm_pin_1, 100)  # Frekuensi PWM 100Hz
pwm2 = GPIO.PWM(pwm_pin_2, 100)
pwm1.start(0)  # Mulai PWM dengan duty cycle 0 (motor berhenti)
pwm2.start(0)

# Fungsi motor
def motor_forward(speed):
    pwm1.ChangeDutyCycle(speed)
    pwm2.ChangeDutyCycle(0)

def motor_stop():
    pwm1.ChangeDutyCycle(0)
    pwm2.ChangeDutyCycle(0)

try:
    while True:
        sensor_state = GPIO.input(sensor_pin)
        
        if sensor_state == GPIO.LOW:  # Sensor mendeteksi objek
            print("Sensor mendeteksi objek, menghentikan konveyor...")
            motor_stop()
            time.sleep(1)  # Beri jeda agar motor benar-benar berhenti
            
            # Kirim permintaan ke server Django untuk menangkap gambar
            response = requests.get("http://localhost:8000/capture_image/")
            if response.status_code == 200:
                print("Gambar berhasil diambil.")
            else:
                print("Gagal mengambil gambar.")

            time.sleep(3)  # Beri waktu untuk pemrosesan gambar
            print("Melanjutkan konveyor...")
            motor_forward(100)  # Hidupkan kembali konveyor
        
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Program dihentikan oleh pengguna.")
    motor_stop()
    pwm1.stop()
    pwm2.stop()
    GPIO.cleanup()
