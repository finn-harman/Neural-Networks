from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from part2_claim_classifier import *
import pickle
import numpy as np
import pandas as pd

def fit_and_calibrate_classifier(classifier, X):
    # DO NOT ALTER THIS FUNCTION
    X_train, X_cal = train_test_split(
        X, train_size=0.85, random_state=0)
    classifier.fit(X_train)

    # This line does the calibration for you
    calibrated_classifier = CalibratedClassifierCV(
        classifier, method='sigmoid', cv='prefit').fit(X_cal)
    return calibrated_classifier


# class for part 3
class PricingModel():
    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY
    def __init__(self, calibrate_probabilities=False):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """
        self.y_mean = None
        self.calibrate = calibrate_probabilities
        self.attributes = None
        # =============================================================
        # READ ONLY IF WANTING TO CALIBRATE
        # Place your base classifier here
        # NOTE: The base estimator must have:
        #    1. A .fit method that takes two arguments, X, y
        #    2. Either a .predict_proba method or a decision
        #       function method that returns classification scores
        #
        # Note that almost every classifier you can find has both.
        # If the one you wish to use does not then speak to one of the TAs
        #
        # If you wish to use the classifier in part 2, you will need
        # to implement a predict_proba for it before use
        # =============================================================
        self.base_classifier = ClaimClassifier() # ADD YOUR BASE CLASSIFIER HERE


    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY TO THE _preprocessor METHOD
    def _preprocessor(self, X_raw):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        X: ndarray
            A clean data set that is used for training and prediction.
        """
        # =============================================================
        # YOUR CODE HERE
        return self.base_classifier._preprocessor(X_raw=X_raw, sampling_type="under") # YOUR CLEAN DATA AS A NUMPY ARRAY

    def fit(self, X_raw, y_raw, claims_raw):
        """Classifier training function.

        Here you will use the fit function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded
        y_raw : ndarray
            A one dimensional array, this is the binary target variable
        claims_raw: ndarray
            A one dimensional array which records the severity of claims

        Returns
        -------
        self: (optional)
            an instance of the fitted model

        """
        claims_raw = claims_raw.to_numpy()
        nnz = np.where(claims_raw != 0)[0]
        self.y_mean = np.mean(claims_raw[nnz])
        # =============================================================
        # REMEMBER TO A SIMILAR LINE TO THE FOLLOWING SOMEWHERE IN THE CODE
        X_raw = X_raw.drop(['id_policy', 'pol_insee_code', 'drv_age2', 'drv_sex2',
                    'drv_age_lic2','vh_model','town_mean_altitude',
                    'town_surface_area', 'population', 'commune_code','canton_code',
                    'city_district_code', 'regional_department_code','vh_make'],axis=1)

        categorical_list = list(X_raw.select_dtypes(include=['object']))

        i = 0
        for feature in categorical_list:
            encoder = sk.preprocessing.LabelBinarizer()
            encoder.fit(X_raw[feature])
            transformed = encoder.transform(X_raw[feature])
            if (len(encoder.classes_) > 2):
                transformed = transformed[:,1:]
            if (len(encoder.classes_) == 2 and encoder.classes_[1] == 'Yes'):
                i += 1
                encoder.classes_[1] = i
            ohe_df = pd.DataFrame(transformed, columns=encoder.classes_[1:], index=X_raw.index)
            X_raw = pd.concat([X_raw, ohe_df], axis=1).drop([feature], axis=1)

        self.attributes = list(X_raw.columns.values)
        self.attributes.remove('claim_amount')
        self.attributes.remove('made_claim')

        #X_clean = self._preprocessor(X_raw)

        # THE FOLLOWING GETS CALLED IF YOU WISH TO CALIBRATE YOUR PROBABILITIES
        if self.calibrate:
            self.base_classifier = fit_and_calibrate_classifier(
                self.base_classifier, X_raw)
        else:
            self.base_classifier.fit(X_raw)


    def predict_claim_probability(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # =============================================================
        # REMEMBER TO A SIMILAR LINE TO THE FOLLOWING SOMEWHERE IN THE CODE
        print(X_raw.shape)

        X_raw = X_raw.drop(['id_policy', 'pol_insee_code', 'drv_age2', 'drv_sex2',
                    'drv_age_lic2','vh_model','town_mean_altitude',
                    'town_surface_area', 'population', 'commune_code','canton_code',
                    'city_district_code', 'regional_department_code','vh_make'],axis=1)

        categorical_list = list(X_raw.select_dtypes(include=['object']))

        i = 0
        for feature in categorical_list:
            encoder = sk.preprocessing.LabelBinarizer()
            encoder.fit(X_raw[feature])
            transformed = encoder.transform(X_raw[feature])
            if (len(encoder.classes_) > 2):
                transformed = transformed[:,1:]
            if (len(encoder.classes_) == 2 and encoder.classes_[1] == 'Yes'):
                i += 1
                encoder.classes_[1] = i
            ohe_df = pd.DataFrame(transformed, columns=encoder.classes_[1:], index=X_raw.index)
            X_raw = pd.concat([X_raw, ohe_df], axis=1).drop([feature], axis=1)

        for attribute in self.attributes:
            if attribute not in X_raw:
                X_raw[attribute] = 0

        X_raw = X_raw[self.attributes]

        return self.base_classifier.predict(X_raw) # return probabilities for the positive class (label 1)

    def predict_premium(self, X_raw):
        """Predicts premiums based on the pricing model.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : numpy.ndarray
            A numpy array, this is the raw data as downloaded

        Returns
        -------
        numpy.ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # =============================================================
        # REMEMBER TO INCLUDE ANY PRICING STRATEGY HERE.
        # For example you could scale all your prices down by a factor
        scale = 0.202
        premium = scale * self.predict_claim_probability(X_raw) * self.y_mean
        premium = premium.squeeze()
        return premium

    def set_premium_scale(self, premiums, train_claim_severity):
        scale = np.sum(train_claim_severity) / np.sum(premiums)
        return scale

    def save_model(self):
        """Saves the class instance as a pickle file."""
        # =============================================================
        with open('part3_pricing_model.pickle', 'wb') as target:
            pickle.dump(self, target)


def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open('part3_pricing_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model