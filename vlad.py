import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import os
import cv2

def extract_descriptors(image_path):
    orb = cv2.ORB_create()
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return descriptors

def extract_descriptors_img(image):
    orb = cv2.ORB_create()
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return descriptors

def compute_vlad(descriptors, centers):
    labels = np.argmin(euclidean_distances(descriptors, centers), axis=1)

    vlad_vector = np.zeros((len(centers), descriptors.shape[1]))

    for i in range(len(centers)):
        vlad_vector[i, :] = np.sum(descriptors[labels == i, :] - centers[i, :], axis=0)

    vlad_vector = vlad_vector.flatten()
    vlad_vector /= np.sqrt(np.sum(vlad_vector**2))

    return vlad_vector

def load_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            path = os.path.join(directory, filename)
            images.append(path)
    return images

def main():
    query_paths = ["query_images/7.jpg", "query_images/132.jpg"]

    database_directory = "images/"
    database_paths = load_images_from_directory(database_directory)

    k = 64

    query_descriptors = [extract_descriptors(path) for path in query_paths]

    data_descriptors = [extract_descriptors(path) for path in database_paths]

    print("Finding Cluster Centers....")
    all_descriptors = np.concatenate(data_descriptors, axis=0)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(all_descriptors)
    cluster_centers = kmeans.cluster_centers_
    print("Found Cluster Centers")

    query_vlads = [compute_vlad(query_descriptor, cluster_centers) for query_descriptor in query_descriptors]

    database_vlads = []
    i = 0
    for path in database_paths:
        print("i : ", i)
        database_descriptor = data_descriptors[i]
        database_vlad = compute_vlad(database_descriptor, cluster_centers)
        database_vlads.append(database_vlad)
        i += 1
    #print(database_vlads[0])
    for i, query_vlad in enumerate(query_vlads):
        similarities = np.dot(database_vlads, query_vlad)
        most_similar_index = np.argmax(similarities)

        # Display the query image and the most similar database image
        query_image = cv2.imread(query_paths[i])
        similar_image = cv2.imread(database_paths[most_similar_index])

        cv2.imshow(f"Query {i + 1}", query_image)
        cv2.imshow(f"Most Similar {i + 1}", similar_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f"Query {i + 1}: Most similar database image is {most_similar_index + 1} with similarity {similarities[most_similar_index]}") 

if __name__ == "__main__":
    main()
