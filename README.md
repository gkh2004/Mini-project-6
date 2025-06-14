# Mini-project-6
ADVANCE PYTHON ASSIGNMENT
import random
import string

with open("input.txt", "w") as f:
    for _ in range(1000):
        line = ''.join(random.choices(string.ascii_letters + string.digits, k=80))
        f.write(line + '\n')

import random
import string
import os

target_size = 5 * 1024 * 1024  # 5 MB in bytes
line_length = 80  # characters per line (excluding newline)

with open("input.txt", "w") as f:
    while f.tell() < target_size:
        line = ''.join(random.choices(string.ascii_letters + string.digits, k=line_length))
        f.write(line + '\n')

print("File 'input.txt' created with size:", os.path.getsize("input.txt"), "bytes")

import random
import string
import os

def generate_random_file(filename, size_in_bytes, line_length=80):
    with open(filename, "w") as f:
        while f.tell() < size_in_bytes:
            line = ''.join(random.choices(string.ascii_letters + string.digits, k=line_length))
            f.write(line + '\n')

file_count = 10
target_size = 5 * 1024 * 1024  # 5 MB in bytes

for i in range(1, file_count + 1):
    filename = f"input_{i}.txt"
    generate_random_file(filename, target_size)
    print(f"{filename} created, size: {os.path.getsize(filename)} bytes")

    import random
import string
import os

def generate_large_file(filename, size_in_bytes, line_length=100):
    with open(filename, "w") as f:
        bytes_written = 0
        while bytes_written < size_in_bytes:
            line = ''.join(random.choices(string.ascii_letters + string.digits, k=line_length)) + '\n'
            f.write(line)
            bytes_written += len(line)

# File sizes in GB
file_sizes_gb = [1, 2, 3, 4, 5]

for size_gb in file_sizes_gb:
    size_bytes = size_gb * 1024 * 1024 * 1024  # Convert GB to bytes
    filename = f"random_{size_gb}GB.txt"
    print(f"Creating {filename} ({size_gb} GB)...")
    generate_large_file(filename, size_bytes)
    actual_size = os.path.getsize(filename)
    print(f"Done: {filename} (Size: {actual_size / (1024 ** 3):.2f} GB)")

import os

# File sizes (1GB to 5GB)
file_sizes_gb = [1, 2, 3, 4, 5]

for size_gb in file_sizes_gb:
    input_file = f"random_{size_gb}GB.txt"
    output_file = f"random_{size_gb}GB_upper.txt"

    print(f"Converting {input_file} to uppercase...")

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            outfile.write(line.upper())

    print(f"Done: {output_file} created.")

   import os
from concurrent.futures import ThreadPoolExecutor

def convert_file_to_upper(input_file, output_file):
    print(f"Starting: {input_file} ➜ {output_file}")
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            outfile.write(line.upper())
    print(f"Done: {output_file}")

# File list
file_sizes_gb = [1, 2, 3, 4, 5]
tasks = [
    (f"random_{size}GB.txt", f"random_{size}GB_upper.txt")
    for size in file_sizes_gb
]

# Run conversions in parallel using threads
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(convert_file_to_upper, inp, out) for inp, out in tasks]

    # Wait for all to complete
    for future in futures:
        future.result()  # This will also raise exceptions if any

        
from icrawler.builtin import GoogleImageCrawler

# Create a crawler object
google_crawler = GoogleImageCrawler(storage={'root_dir': 'cat_images'})

# Download 10 cat images
google_crawler.crawl(keyword='cat', max_num=10)

print("Downloaded 10 cat images into 'cat_images' folder.")

from pytube import Search, YouTube
import os

# Create a folder to store videos
os.makedirs("videos", exist_ok=True)

# Search YouTube for "Machine Learning"
search = Search("Machine Learning")
videos = search.results[:10]  # First 10 results

for i, video in enumerate(videos, start=1):
    try:
        yt = YouTube(video.watch_url)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        print(f"Downloading video {i}: {yt.title}")
        stream.download(output_path="videos", filename=f"video_{i}.mp4")
    except Exception as e:
        print(f"Failed to download video {i}: {e}")


from moviepy.editor import VideoFileClip
import os

# Create folder for audio files
os.makedirs("audios", exist_ok=True)

# Convert each video in 'videos/' to audio
for filename in os.listdir("videos"):
    if filename.endswith(".mp4"):
        video_path = os.path.join("videos", filename)
        audio_path = os.path.join("audios", filename.replace(".mp4", ".mp3"))

        print(f"Converting {filename} to audio...")
        try:
            clip = VideoFileClip(video_path)
            clip.audio.write_audiofile(audio_path)
            clip.close()
        except Exception as e:
            print(f"Failed to convert {filename}: {e}")


            pip install pytube moviepy
import os
from pytube import Search, YouTube
from moviepy.editor import VideoFileClip
from concurrent.futures import ThreadPoolExecutor, as_completed

# Prepare directories
os.makedirs("videos", exist_ok=True)
os.makedirs("audios", exist_ok=True)

# Search YouTube for videos
search_query = "Machine Learning"
search_results = Search(search_query).results[:100]

# ----- Step 1: Download Function -----
def download_video(index, url):
    try:
        yt = YouTube(url)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        filename = f"video_{index}.mp4"
        path = os.path.join("videos", filename)
        stream.download(output_path="videos", filename=filename)
        print(f"✅ Downloaded: {filename}")
        return path  # Return video path
    except Exception as e:
        print(f"❌ Failed to download video {index}: {e}")
        return None

# ----- Step 2: Convert to Audio Function -----
def convert_to_audio(video_path):
    try:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = os.path.join("audios", base_name + ".mp3")
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path)
        clip.close()
        print(f"🎧 Converted: {base_name}.mp3")
    except Exception as e:
        print(f"❌ Failed to convert {video_path}: {e}")

# ----- Step 3: Threaded Pipeline -----
def pipeline_task(index, url):
    video_path = download_video(index, url)
    if video_path:
        convert_to_audio(video_path)

# ----- Step 4: Run Multithreaded Pipeline -----
def run_pipeline():
    print("🚀 Starting download + conversion pipeline for 100 videos...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(pipeline_task, idx+1, video.watch_url) for idx, video in enumerate(search_results)]
        for future in as_completed(futures):
            future.result()  # Handle exceptions

run_pipeline()


pip install icrawler pillow
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from icrawler.builtin import GoogleImageCrawler

# 📁 Create folders
os.makedirs("dog_images_original", exist_ok=True)
os.makedirs("dog_images_resized", exist_ok=True)

# 📥 Step 1: Download Images (not multithreaded because icrawler handles batching)
def download_images():
    print("🔍 Starting image download...")
    google_crawler = GoogleImageCrawler(storage={'root_dir': 'dog_images_original'})
    google_crawler.crawl(keyword='dog', max_num=500)
    print("✅ Download complete.")

# 📏 Step 2: Resize Function
def resize_image(image_path):
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            resized_img = img.resize((w // 2, h // 2))
            save_path = os.path.join("dog_images_resized", os.path.basename(image_path))
            resized_img.save(save_path)
            print(f"📐 Resized: {os.path.basename(image_path)}")
    except Exception as e:
        print(f"❌ Failed to resize {image_path}: {e}")

# 🚀 Step 3: Multithreaded Resizing
def resize_all_images_multithreaded():
    print("🚀 Starting resizing with threads...")
    images = [os.path.join("dog_images_original", f) for f in os.listdir("dog_images_original") if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(resize_image, image) for image in images]
        for future in as_completed(futures):
            future.result()  # To catch any exceptions

# 🧵 Step 4: Run the pipeline
def run_pipeline():
    download_images()
    resize_all_images_multithreaded()
    print("🏁 Pipeline complete.")

# ▶️ Execute the pipeline
run_pipeline()


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Step 1: Create dataset
np.random.seed(42)
data = np.random.randint(1, 201, size=(100, 30))
df = pd.DataFrame(data, columns=[f"Col_{i+1}" for i in range(30)])

# Step (i): Replace values [10,60] with NA, count rows with NA
df_masked = df.mask(df.between(10, 60))
rows_with_na = df_masked.isna().any(axis=1).sum()
print(f"Number of rows with missing values: {rows_with_na}")

# Step (ii): Replace NA with column mean
df_filled = df_masked.fillna(df_masked.mean())

# Step (iii): Pearson correlation heatmap and select cols with max corr <= 0.7
corr_matrix = df_filled.corr(method='pearson')
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False)
plt.title('Pearson Correlation Heatmap')
plt.show()

cols_to_keep = [col for col in corr_matrix.columns if corr_matrix[col].drop(col).abs().max() <= 0.7]
print(f"Columns with max correlation <= 0.7: {cols_to_keep}")

# Step (iv): Normalize to [0, 10]
scaler = MinMaxScaler(feature_range=(0, 10))
df_normalized = pd.DataFrame(scaler.fit_transform(df_filled), columns=df_filled.columns)

# Step (v): Replace values <= 0.5 with 1, else 0
df_binary = df_normalized.applymap(lambda x: 1 if x <= 0.5 else 0)

# Show first few rows of final binary dataframe
print(df_binary.head())



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score

# Step 1: Create dataset
np.random.seed(42)

cols_1_4 = np.random.uniform(-10, 10, size=(500, 4))
cols_5_8 = np.random.uniform(10, 20, size=(500, 4))
cols_9_10 = np.random.uniform(-100, 100, size=(500, 2))

data = np.hstack((cols_1_4, cols_5_8, cols_9_10))
df = pd.DataFrame(data, columns=[f"Col_{i+1}" for i in range(10)])

# Standardize the dataset for clustering
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
inertia = []
sil_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(df_scaled, kmeans.labels_))

# Plot Inertia (Elbow Method)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertia, marker='o')
plt.title("K-Means: Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")

# Plot Silhouette Score
plt.subplot(1, 2, 2)
plt.plot(K_range, sil_scores, marker='o', color='green')
plt.title("K-Means: Silhouette Score")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.tight_layout()
plt.show()

# Best k (can use max silhouette)
optimal_k = K_range[np.argmax(sil_scores)]
print(f"✅ Optimal k for K-Means based on Silhouette Score: {optimal_k}")
# Compute linkage matrix
linked = linkage(df_scaled, method='ward')

# Plot dendrogram
plt.figure(figsize=(12, 6))
dendrogram(linked, truncate_mode='lastp', p=30, leaf_rotation=90., leaf_font_size=10.)
plt.title("Hierarchical Clustering Dendrogram (truncated)")
plt.xlabel("Sample index or (cluster size)")
plt.ylabel("Distance")
plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Generate dataset
np.random.seed(42)
data = np.random.uniform(-100, 100, size=(600, 15))
df = pd.DataFrame(data, columns=[f"Col_{i+1}" for i in range(15)])

# (i) Scatter plot between Column 5 and Column 6
plt.figure(figsize=(6, 5))
plt.scatter(df['Col_5'], df['Col_6'], color='blue', alpha=0.6)
plt.title('Scatter Plot: Column 5 vs Column 6')
plt.xlabel('Col_5')
plt.ylabel('Col_6')
plt.grid(True)
plt.show()

# (ii) Histograms of all columns in a single figure
df.hist(figsize=(16, 10), bins=20, edgecolor='black', grid=False)
plt.suptitle('Histograms of All Columns', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()

# (iii) Box plots of all columns in a single figure
plt.figure(figsize=(14, 6))
sns.boxplot(data=df, orient="v")
plt.title("Box Plots of All Columns")
plt.xlabel("Columns")
plt.ylabel("Values")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


import numpy as np
import pandas as pd
from scipy import stats

# Step 1: Create dataset (500 rows, 5 columns, values between [5, 10])
np.random.seed(42)
data = np.random.uniform(5, 10, size=(500, 5))
df = pd.DataFrame(data, columns=[f"Col_{i+1}" for i in range(5)])

# (i) Perform one-sample t-test (H0: mean = 7.5)
print("One-Sample t-Test (H0: mean = 7.5):")
for col in df.columns:
    t_stat, p_val = stats.ttest_1samp(df[col], popmean=7.5)
    print(f"{col}: t-stat = {t_stat:.4f}, p-value = {p_val:.4f}")

print("\n" + "-"*50)

# (ii) Perform Wilcoxon Signed Rank Test (H0: median = 7.5)
# Note: Wilcoxon test requires symmetric paired differences — non-parametric version of t-test
print("Wilcoxon Signed Rank Test (H0: median = 7.5):")
for col in df.columns:
    try:
        w_stat, p_val = stats.wilcoxon(df[col] - 7.5)
        print(f"{col}: W = {w_stat:.4f}, p-value = {p_val:.4f}")
    except Exception as e:
        print(f"{col}: Wilcoxon test failed – {e}")

print("\n" + "-"*50)

# (iii) Two-sample t-test and Wilcoxon Rank Sum Test between Column 3 and Column 4
col3 = df['Col_3']
col4 = df['Col_4']

# Two-sample t-test (independent)
t_stat, p_val = stats.ttest_ind(col3, col4, equal_var=False)
print("Two-Sample t-Test (Col_3 vs Col_4):")
print(f"t-stat = {t_stat:.4f}, p-value = {p_val:.4f}")

# Wilcoxon Rank Sum Test (Mann-Whitney U test)
u_stat, p_val = stats.mannwhitneyu(col3, col4, alternative='two-sided')
print("Wilcoxon Rank Sum Test (Col_3 vs Col_4):")
print(f"U-stat = {u_stat:.4f}, p-value = {p_val:.4f}")



