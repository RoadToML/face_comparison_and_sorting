import os
import shutil

import cv2
import face_recognition

from PIL import Image

def compare_faces(known_face_dir, unknown_faces_dir, transfer_dir, size = 300, tolerance = 0.6):
  """
  this function is responsible for comparing faces and returning the pictures where that face is present (given tolerance)
  into a new dir.

  INPUT: {
      known_face_dir: This positional argument should be a path to the dir where a known image exists.
                      This directory should contain 1 face for best result.

      unknown_faces_dir: This positional argument should be a path to the dir where you want to compare the face from known_face_dir.
                         This dir can have many images.

      transfer_dir: this positional argument should be the path to the dir where all matching images should be transfered to.

      size: This default argument can be specified to resize the images when they are opened as they may provide better accuracy in recognising faces.
            (takes in int values)

      tolerance: This default argument can be specified to measure how similar the compared faces are (i.e. 60%/ 0.6). lower the better.
                (takes in float values)
  }

  OUTPUT: {
        this function populates the transfer_dir with all matching imagees.
  }
  """
  # open all images
  for i in os.listdir(known_face_dir):
    image = cv2.imread(os.path.join(known_face_dir, i))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size,size))

    print(image.shape)

    # creates encoding for all faces in the original image (returns a list)
    face_image_encodings = face_recognition.face_encodings(image)

    for single_encoding in face_image_encodings:

      # for loop to iterate over all images in the unknown_faces_dir
      for x in os.listdir(unknown_faces_dir):

        new_image = cv2.imread(os.path.join(unknown_faces_dir, x))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        new_image = cv2.resize(new_image, (size,size))

        # encoding for faces in the unknown_faces_dir (returns a list)
        new_face_image_encodings = face_recognition.face_encodings(new_image)

        for each_encoding in new_face_image_encodings:

          # if cond to compare faces - currently facing an error where u cant have 2 encoding apparently.
          if face_recognition.compare_faces([single_encoding], each_encoding, tolerance = tolerance)[0]:

            shutil.copyfile(os.path.join(unknown_faces_dir, x), os.path.join(transfer_dir, x))
            print('copied')
            break

          else:
            print('no face found in image or below threshold (matching %)')


if __name__ == '__main__':

    image_path = '/path/to/file/image_path'
    all_image_path = '/path/to/file/all_image_path'
    move_path = '/path/to/file/move_path'

    compare_faces(image_path, all_image_path, move_path, size = 300, tolerance = 0.55)
