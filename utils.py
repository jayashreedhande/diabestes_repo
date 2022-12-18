import pickle 
import json 
import config_file
import numpy  as np 

class DiabetesData():

    def __init__(self,user_data):
        self.model_file_path = 'diabetes.pkl'
        self.user_data = user_data

    def load_saved_data(self):
        with open(self.model_file_path,'rb') as f:
            self.model = pickle.load(f)
        
        # with open('project_data.json','r') as f:
        #     self.proj_data = json.load(f)
        
    def get_predicted_class(self):

        self.load_saved_data()

        Glucose = eval(self.user_data['Glucose'])
        BloodPressure = eval(self.user_data['BloodPressure'])
        SkinThickness = eval(self.user_data['SkinThickness'])
        Insulin = eval(self.user_data['Insulin'])
        BMI= eval(self.user_data['BMI'])
        DiabetesPedigreeFunction = eval(self.user_data['DiabetesPedigreeFunction'])
        Age = eval(self.user_data['Age'])

        test_array= np.array([Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

        print(test_array)

        pred_Class = self.model.predict([test_array])[0]
        print('pred_class:',pred_Class)
        return pred_Class 

if __name__ == "__main__":
    diab = DiabetesData()
    diab