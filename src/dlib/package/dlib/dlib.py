##
# Copyright 2018, Ammar Ali Khan
# Licensed under MIT.
# Since: v1.0.0
##

import dlib
from imutils import face_utils
from src.dlib.package.config import application


##
# Dlib class
# This class is a wrapper for Digital Library (Dlib)
#
# @see: http://dlib.net/
##
class Dlib:

    def __init__(self):
        print('[INFO] Dlib - Initialising...')
        self._frontal_face_detector = dlib.get_frontal_face_detector()
        self._shape_predictor_68_face_landmarks = dlib.shape_predictor(
            application.SHAPE_PREDICTOR_68_FACE_LANDMARKS_PATH
        )

    ##
    # Method frontal_face_detector()
    # Method to return Dlib frontal_face_detector
    #
    # @param frame - image frame
    # @param up_sample - enlarge image for better capture (0-1)
    #
    # @return Array of detection(s)
    ##
    def frontal_face_detector(self, frame, up_sample):
        return self._frontal_face_detector(frame, up_sample)

    ##
    # Method shape_predictor_68_face_landmarks()
    # Method to return Dlib shape_predictor_68_face_landmarks
    #
    # @param frame - image frame
    # @param coordinate - face detection coordinates
    #
    # @return Array of landmark(s)
    ##
    def shape_predictor_68_face_landmarks(self, frame, coordinates):
        landmarks = self._shape_predictor_68_face_landmarks(frame, coordinates)
        return face_utils.shape_to_np(landmarks)

