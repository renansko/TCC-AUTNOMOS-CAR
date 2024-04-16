import time

# Define the safe distance in centimeters
safe_distance = 30  # Change this value to your desired safe distance

# Define the sensor
sensor = HCSR04()  # Replace HCSR04 with your distance sensor model

while True:
    # Read the distance from the sensor
    distance_cm = sensor.distance_cm()

    # Check if the distance is less than the safe distance
    if distance_cm < safe_distance:
        # If the distance is less than the safe distance, trigger an error
        print("Error: Object detected within the safe distance!")
    else:
        # If the distance is greater than the safe distance, print a safe message
        print("Safe distance maintained.")

    # Wait for a short period of time before checking again
    time.sleep(0.1)