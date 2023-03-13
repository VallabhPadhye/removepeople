import streamlit as st

import cv2

import numpy as np

# Define a function to remove people from an image

def remove_people(image_path, selected_people):

    # Load the image

    img = cv2.imread(image_path)

    # Convert the image to grayscale

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained Haar cascade classifier for detecting people

    cascade_path = "haarcascade_fullbody.xml"

    cascade = cv2.CascadeClassifier(cascade_path)

    # Detect people in the image

    people_rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Remove selected people from the image

    for i, (x, y, w, h) in enumerate(people_rects):

        if i in selected_people:

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

        # Load the image

        img = cv2.imread(image_path)

        # Convert the image to RGB format for display in Streamlit

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Display the original image

        st.image(img, caption="Original Image", width=300)

        # Load the pre-trained Haar cascade classifier for detecting people

        cascade_path = "haarcascade_fullbody.xml"

        cascade = cv2.CascadeClassifier(cascade_path)

        # Detect people in the image

        gray = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)

        people_rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Display a checkbox for each person detected in the image

        selected_people = []

        for i, (x, y, w, h) in enumerate(people_rects):

            checkbox = st.checkbox(f"Person {i+1} ({w}x{h})", key=i)

            if checkbox:

                selected_people.append(i)

        # Remove selected people from the image

        removed_image = remove_people(image_path, selected_people)

        # Convert the removed image to RGB format for display in Streamlit

        removed_image = cv2.cvtColor(removed_image, cv2.COLOR_BGR2RGB)

        # Display the modified image

        st.image(removed_image, caption="Removed People", width=300)

# Run the Streamlit app

if __name__ == '__main__':

    app()

