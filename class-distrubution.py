import matplotlib.pyplot as plt

# Provided class distribution and weightage data
class_names = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy", 
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy", 
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", 
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot", 
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy", 
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy", 
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight", 
    "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew", 
    "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight", 
    "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", 
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", 
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

image_counts = [
    504, 496, 220, 1316, 1202, 842, 684, 410, 953, 788, 929, 944, 1107, 861, 339, 4405, 
    1838, 288, 797, 1183, 800, 800, 121, 297, 4072, 1468, 887, 364, 1702, 800, 1527, 
    761, 1417, 1341, 1123, 4286, 299, 1273
]

weights = [
    1.71, 1.74, 3.91, 0.65, 0.72, 1.02, 1.26, 2.10, 0.90, 1.09, 0.93, 0.91, 0.78, 1.00, 2.54, 
    0.20, 0.47, 2.99, 1.08, 0.73, 1.08, 1.08, 7.12, 2.90, 0.21, 0.59, 0.97, 2.37, 0.51, 1.08, 
    0.56, 1.13, 0.61, 0.64, 0.77, 0.20, 2.88, 0.68
]

# Create a figure and axes for the subplots
fig, ax = plt.subplots(1, 2, figsize=(15, 7))

# Bar chart for the number of images per class
ax[0].barh(class_names, image_counts, color='skyblue')
ax[0].set_xlabel('Number of Images')
ax[0].set_ylabel('Class')
ax[0].set_title('Number of Images per Class')

# Bar chart for the weightage per class
ax[1].barh(class_names, weights, color='lightgreen')
ax[1].set_xlabel('Weightage')
ax[1].set_ylabel('Class')
ax[1].set_title('Weightage per Class')

# Adjust layout to make room for the titles and labels
plt.tight_layout()

# Show the plots
plt.show()
