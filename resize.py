import os
import time 
import threading
from PIL import Image
from tqdm import tqdm
import psutil

# Resize dimensions
target_size = (640, 640)

#monitor system performance
def monitor_performance(interval = 10):
    print(f"\nMonitoring device performance every{interval} seconds)")
    while True:
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        print(f"CPU Usage: {cpu_usage}%, Memory Usage: {memory_info.percent}%")
        time.sleep(interval)
        
    
#Running the performance monitoring in a separate thread
monitor_thread = threading.Thread(target=monitor_performance, daemon=True)
monitor_thread.start()

#input folder path
input_folder = "./raw"
output_folder = "./resized"



# Loop through each file in the input folder with a progress bar
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png','.jpg','.jpeg'))]
print("Resizing images:")

for filename in tqdm(image_files, desc="Progress", unit="image", colour="green"):
        try:
            # Open image
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)

            #Resize image using Image.Resampling.LANCZOS instead of Image.ANTIALIAS
            img_resized = img.resize(target_size, Image.Resampling.LANCZOS) 

            # Save resized image to output folder
            output_path = os.path.join(output_folder, filename)
            img_resized.save(output_path)
        except Exception as e:
            print(f'Error resizing image: {filename}')

print(f'Resized and saved: {output_path}')
        