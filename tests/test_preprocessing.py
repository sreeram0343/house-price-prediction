import unittest
import pandas as pd
import numpy as np
from src.preprocessing import engineer_features

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        # Sample raw dataframe setup matching dataset format
        self.sample_data = pd.DataFrame({
            'id': [1, 2],
            'Date': [42491, 42734],
            'number of bedrooms': [3, 4],
            'number of bathrooms': [2, 2.5],
            'living area': [2000, 3000],
            'lot area': [5000, 6000],
            'number of floors': [1.5, 2],
            'waterfront present': [0, 1],
            'number of views': [0, 2],
            'condition of the house': [3, 4],
            'grade of the house': [7, 8],
            'Area of the house(excluding basement)': [2000, 2500],
            'Area of the basement': [0, 500],
            'Built Year': [1990, 2010],
            'Renovation Year': [0, 2015],
            'Postal Code': [122032, 122032],
            'Lattitude': [52.8, 52.9],
            'Longitude': [-114.4, -114.5],
            'living_area_renov': [2000, 3200],
            'lot_area_renov': [5000, 5800],
            'Number of schools nearby': [2, 3],
            'Distance from the airport': [65, 70],
            'Price': [1000000, 1500000]
        })

    def test_engineer_features(self):
        df_out = engineer_features(self.sample_data)
        
        # Check that ID and Date are dropped
        self.assertNotIn('id', df_out.columns)
        self.assertNotIn('Date', df_out.columns)
        
        # Check newly engineered columns
        self.assertIn('house_age', df_out.columns)
        self.assertIn('is_renovated', df_out.columns)
        self.assertIn('years_since_last_mod', df_out.columns)
        self.assertIn('total_rooms', df_out.columns)
        self.assertIn('bed_bath_ratio', df_out.columns)
        self.assertIn('living_lot_ratio', df_out.columns)
        self.assertIn('has_basement', df_out.columns)
        
        # Assert specific values
        # house_age for 1990 (with base year 2016) = 26
        # house_age for 2010 = 6
        self.assertEqual(df_out['house_age'].iloc[0], 26)
        self.assertEqual(df_out['house_age'].iloc[1], 6)
        
        # is_renovated for Renovation Year 0 is 0, for 2015 is 1
        self.assertEqual(df_out['is_renovated'].iloc[0], 0)
        self.assertEqual(df_out['is_renovated'].iloc[1], 1)
        
        # total_rooms: bedrooms + bathrooms
        self.assertEqual(df_out['total_rooms'].iloc[0], 5.0)
        self.assertEqual(df_out['total_rooms'].iloc[1], 6.5)
        
        # has_basement: basement 0 -> 0, basement 500 -> 1
        self.assertEqual(df_out['has_basement'].iloc[0], 0)
        self.assertEqual(df_out['has_basement'].iloc[1], 1)

if __name__ == '__main__':
    unittest.main()
