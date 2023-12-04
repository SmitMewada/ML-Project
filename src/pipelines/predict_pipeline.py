import sys 
import pandas as pd

from src.exception import CustomException
from src.utils import load_object

class PredictPipline:
    def __init__(self):
        pass
    
    
class CustomData():
    def __init__(self,
                 gender: str,
                 race: str,
                 parental_level_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        
        self.gender = gender
        self.race = race
        self.parental_level_education = parental_level_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        
    def get_data_as_data_frame(self):
        
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race": [self.race],
                "parental_level_education": [self.parental_level_education],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
        
        
        