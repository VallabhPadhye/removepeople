import streamlit as st

import cv2

# Define a function to remove people from an image

def remove_people(image_path):

    # Load the image

    img = cv2.imread(image_path)

    # Convert the image to grayscale

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained Haar cascade classifier for detecting people

    cascade_path = "haarcascade_fullbody.xml"

    cascade = cv2.CascadeClassifier(cascade_path)

    # Detect people in the image

    people_rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Replace each person in the image with the average color of the surrounding pixels

    for (x, y, w, h) in people_rects:

        # Get the average color of the surrounding pixels

        avg_color = cv2.mean(img[y-5:y+h+5, x-5:x+w+5])[0:3]

        # Replace the person with the average color

        cv2.rectangle(img, (x, y), (x+w, y+h), avg_color, -1)

    # Return the modified image

    return img

# Define the Streamlit app

def app():

    # Set the page title

    st.set_page_config(page_title="Remove People from Image")

    # Display the title

    st.title("Remove People from Image")

    # Get the image file from the user

    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # If the user has uploaded an image, process it and display the result

    if image_file is not None:

        # Read the image file

        image_bytes = image_file.read()

        image_path = "uploaded_image.jpg"

        with open(image_path, "wb") as f:

            f.write(image_bytes)

        # Remove people from the image

        removed_image = remove_people(image_path)

        # Display the original image and the modified image side by side

        st.image([cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), cv2.cvtColor(removed_image, cv2.COLOR_BGR2RGB)], caption=["Original Image", "Removed People"], width=300)

