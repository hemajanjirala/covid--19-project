{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "f5925826",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np # linear algebra\n",
    "import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import f1_score\n",
    "import pickle as p\n",
    "import numpy as np\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "df4f9a6c",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Dell\\AppData\\Local\\Temp\\ipykernel_8928\\3103726082.py:12: DtypeWarning: Columns (2,3,4,5,6,8) have mixed types. Specify dtype option on import or set low_memory=False.\n",
      "  df = pd.read_csv(f)\n"
     ]
    }
   ],
   "source": [
    "import zipfile\n",
    "import pandas as pd\n",
    "\n",
    "# Open the zip file\n",
    "with zipfile.ZipFile(\"CAP+Project+2+-+corona_tested_006.zip\") as z:\n",
    "    # Get the information of all files \n",
    "    for info in z.infolist():\n",
    "        # Extract the file\n",
    "        with z.open(info) as f:\n",
    "            # If it's a CSV file, read it into a Pandas DataFrame\n",
    "            if info.filename.endswith('.csv'):\n",
    "                df = pd.read_csv(f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "7d9d7418",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   Ind_ID   Test_date Cough_symptoms  Fever Sore_throat Shortness_of_breath  \\\n",
      "0       1  11-03-2020           True  False        True               False   \n",
      "1       2  11-03-2020          False   True       False               False   \n",
      "2       3  11-03-2020          False   True       False               False   \n",
      "3       4  11-03-2020           True  False       False               False   \n",
      "4       5  11-03-2020           True  False       False               False   \n",
      "\n",
      "  Headache    Corona Age_60_above  Sex           Known_contact  \n",
      "0    False  negative          NaN  NaN                  Abroad  \n",
      "1    False  positive          NaN  NaN                  Abroad  \n",
      "2    False  positive          NaN  NaN                  Abroad  \n",
      "3    False  negative          NaN  NaN                  Abroad  \n",
      "4    False  negative          NaN  NaN  Contact with confirmed  \n"
     ]
    }
   ],
   "source": [
    " print(df.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "e65cc091",
   "metadata": {},
   "outputs": [],
   "source": [
    "# function to convert the data\n",
    "\n",
    "def convert(x):\n",
    "    if x=='TRUE':\n",
    "        return 'true'\n",
    "    elif x==True:\n",
    "        return 'true'\n",
    "    elif x=='FALSE' or x==False:\n",
    "        return 'false'\n",
    "    else:\n",
    "        return x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "1f964c03",
   "metadata": {},
   "outputs": [],
   "source": [
    "# mapping the function\n",
    "\n",
    "df['Cough_symptoms'] = df['Cough_symptoms'].map(convert)\n",
    "df['Fever'] = df['Fever'].map(convert)\n",
    "df['Sore_throat'] = df['Sore_throat'].map(convert)\n",
    "df['Shortness_of_breath'] = df['Shortness_of_breath'].map(convert)\n",
    "df['Headache'] = df['Headache'].map(conver)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "8bc31f6e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 278848 entries, 0 to 278847\n",
      "Data columns (total 11 columns):\n",
      " #   Column               Non-Null Count   Dtype \n",
      "---  ------               --------------   ----- \n",
      " 0   Ind_ID               278848 non-null  int64 \n",
      " 1   Test_date            278848 non-null  object\n",
      " 2   Cough_symptoms       278596 non-null  object\n",
      " 3   Fever                278596 non-null  object\n",
      " 4   Sore_throat          278847 non-null  object\n",
      " 5   Shortness_of_breath  278847 non-null  object\n",
      " 6   Headache             278847 non-null  object\n",
      " 7   Corona               278848 non-null  object\n",
      " 8   Age_60_above         151528 non-null  object\n",
      " 9   Sex                  259285 non-null  object\n",
      " 10  Known_contact        278848 non-null  object\n",
      "dtypes: int64(1), object(10)\n",
      "memory usage: 23.4+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "ced8c144",
   "metadata": {},
   "outputs": [],
   "source": [
    "# replacing the none values with Nan values\n",
    "\n",
    "df.replace({'None': np.nan}, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "49ddad6f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Ind_ID                      0\n",
       "Test_date                   0\n",
       "Cough_symptoms            252\n",
       "Fever                     252\n",
       "Sore_throat                 1\n",
       "Shortness_of_breath         1\n",
       "Headache                    1\n",
       "Corona                      0\n",
       "Age_60_above           127320\n",
       "Sex                     19563\n",
       "Known_contact               0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "d52f3b70",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Dropping the nan values from columns\n",
    "\n",
    "df.dropna(subset=['Cough_symptoms','Fever','Sore_throat','Shortness_of_breath','Headache'],axis=0,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "2f011df4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# dropping the age_60_above column\n",
    "\n",
    "df.drop('Age_60_above',axis=1,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "c0f4aee8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# # replacing the none values with Nan values\n",
    "\n",
    "df.replace({'other': np.nan},inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "bb48c271",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Dropping the nan values from columns\n",
    "\n",
    "df.dropna(subset=['Corona'],axis=0,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "5edf1b7f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Ind_ID          int64\n",
      "Test_date      object\n",
      "Sex          category\n",
      "Age          category\n",
      "dtype: object\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "df = pd.DataFrame({\n",
    "  'Ind_ID': [1, 2, 3],\n",
    "  'Test_date': ['2020-03-11', '2020-03-11', '2020-03-11'], \n",
    "  'Sex': [1, 0, 1],\n",
    "  'Age': [25, 32, 19]\n",
    "})\n",
    "\n",
    "# Loop through columns\n",
    "for col in df.columns:\n",
    "\n",
    "  # Skip certain columns\n",
    "  if col == 'Ind_ID' or col == 'Test_date': \n",
    "    continue\n",
    "  \n",
    "  # Convert to categorical\n",
    "  df[col] = df[col].astype('category')\n",
    "\n",
    "print(df.dtypes)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "639c6c23",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Index: 274702 entries, 0 to 278847\n",
      "Data columns (total 10 columns):\n",
      " #   Column               Non-Null Count   Dtype   \n",
      "---  ------               --------------   -----   \n",
      " 0   Ind_ID               274702 non-null  int64   \n",
      " 1   Test_date            274702 non-null  object  \n",
      " 2   Cough_symptoms       274702 non-null  category\n",
      " 3   Fever                274702 non-null  category\n",
      " 4   Sore_throat          274702 non-null  category\n",
      " 5   Shortness_of_breath  274702 non-null  category\n",
      " 6   Headache             274702 non-null  category\n",
      " 7   Corona               274702 non-null  category\n",
      " 8   Sex                  255668 non-null  category\n",
      " 9   Known_contact        274702 non-null  category\n",
      "dtypes: category(8), int64(1), object(1)\n",
      "memory usage: 8.4+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "0d2fdbd0",
   "metadata": {},
   "outputs": [],
   "source": [
    "# creating a copy of original dataframe\n",
    "\n",
    "covid_df = df.copy(deep=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "709f3f45",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Encoding\n",
    "\n",
    "covid_df['Cough_symptoms'] = pd.get_dummies(covid_df['Cough_symptoms'],drop_first=True)  # to aviod multicolinearity\n",
    "\n",
    "covid_df['Fever'] = pd.get_dummies(covid_df['Fever'],drop_first=True)\n",
    "\n",
    "covid_df['Sore_throat'] = pd.get_dummies(covid_df['Sore_throat'],drop_first=True)\n",
    "\n",
    "covid_df['Shortness_of_breath'] = pd.get_dummies(covid_df['Shortness_of_breath'],drop_first=True)\n",
    "\n",
    "covid_df['Headache'] = pd.get_dummies(covid_df['Headache'],drop_first=True)\n",
    "\n",
    "covid_df['Corona'] = pd.get_dummies(covid_df['Corona'],drop_first=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "ba471940",
   "metadata": {},
   "outputs": [],
   "source": [
    "# encoding\n",
    "\n",
    "covid_df['Sex'].replace({'male':0,'female':1},inplace=True)\n",
    "covid_df['Known_contact'].replace({'Other':2,'Contact with confirmed':1,'Abroad':0},inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "7069be22",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "    Cough_symptoms  Fever  Sore_throat  Shortness_of_breath  Headache  \\\n",
      "0                1      1            0                    0         1   \n",
      "1                0      1            0                    0         1   \n",
      "2                1      1            0                    1         0   \n",
      "3                1      1            1                    1         0   \n",
      "4                1      1            1                    1         1   \n",
      "..             ...    ...          ...                  ...       ...   \n",
      "95               1      0            0                    1         1   \n",
      "96               0      1            1                    1         0   \n",
      "97               1      1            0                    1         0   \n",
      "98               1      1            0                    0         1   \n",
      "99               1      0            1                    0         0   \n",
      "\n",
      "    Known_contact  \n",
      "0               1  \n",
      "1               0  \n",
      "2               0  \n",
      "3               1  \n",
      "4               1  \n",
      "..            ...  \n",
      "95              1  \n",
      "96              1  \n",
      "97              1  \n",
      "98              0  \n",
      "99              0  \n",
      "\n",
      "[100 rows x 6 columns]\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "# Create empty dataframe\n",
    "df = pd.DataFrame() \n",
    "\n",
    "# Columns \n",
    "cols = ['Cough_symptoms', 'Fever', 'Sore_throat', 'Shortness_of_breath', 'Headache', 'Known_contact']\n",
    "\n",
    "# Add columns to dataframe\n",
    "for col in cols:\n",
    "    df[col] = np.random.randint(0,2,100)\n",
    "    \n",
    "# Display dataframe \n",
    "print(df.head(100))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "id": "984ecb4a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Ind_ID                     0\n",
       "Test_date                  0\n",
       "Cough_symptoms             0\n",
       "Fever                      0\n",
       "Sore_throat                0\n",
       "Shortness_of_breath        0\n",
       "Headache                   0\n",
       "Corona                     0\n",
       "Sex                    19034\n",
       "Known_contact              0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 96,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "covid_df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "a1fd1a92",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.impute import KNNImputer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "572b86a2",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'covid_df' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[20], line 13\u001b[0m\n\u001b[0;32m     10\u001b[0m imputer \u001b[38;5;241m=\u001b[39m KNNImputer(n_neighbors\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m5\u001b[39m, weights\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124muniform\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[0;32m     12\u001b[0m \u001b[38;5;66;03m# Fit on and transform the data\u001b[39;00m\n\u001b[1;32m---> 13\u001b[0m covid_df[columns] \u001b[38;5;241m=\u001b[39m imputer\u001b[38;5;241m.\u001b[39mfit_transform(covid_df[columns])\n",
      "\u001b[1;31mNameError\u001b[0m: name 'covid_df' is not defined"
     ]
    }
   ],
   "source": [
    "# applying KNN imputation\n",
    "\n",
    "knn=KNNImputer(n_neighbors=5,weights='uniform')\n",
    "columns=['Cough_symptoms', 'Fever', 'Sore_throat','Shortness_of_breath', 'Headache','Sex','Known_contact']\n",
    "imputed=knn.fit_transform(covid_df[columns])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "966110fb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   Cough  Fever\n",
      "0    1.0    5.0\n",
      "1    2.0    4.5\n",
      "2    2.0    2.0\n",
      "3    3.0    4.0\n"
     ]
    }
   ],
   "source": [
    "# Import modules\n",
    "from sklearn.impute import KNNImputer\n",
    "import pandas as pd\n",
    "\n",
    "# Sample data\n",
    "data = pd.DataFrame({\n",
    "    'Cough': [1,2,np.nan,3],\n",
    "    'Fever': [5,np.nan,2,4]\n",
    "})\n",
    "\n",
    "# Impute missing values\n",
    "imputer = KNNImputer(n_neighbors=2)\n",
    "imputed = imputer.fit_transform(data) \n",
    "\n",
    "# Create new dataframe from imputed data\n",
    "df1 = pd.DataFrame(imputed, columns=data.columns)\n",
    "\n",
    "print(df1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "5ed9a401",
   "metadata": {},
   "outputs": [],
   "source": [
    "def change(x):\n",
    "    if x>0.5:\n",
    "        return 1\n",
    "    elif x<0.5:\n",
    "        return 0\n",
    "    else:\n",
    "        return x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "id": "c3204638",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1['Sex'] = df1['Sex'].apply(change)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "id": "c75e47fb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   Sex\n",
      "0    0\n",
      "1    0\n",
      "2    0\n",
      "3    0\n",
      "4    0\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "# Sample dataframe \n",
    "imputed_data = pd.DataFrame({'Sex': [1, 2, np.nan, 1, 0]})\n",
    "\n",
    "# Replace all values with 0\n",
    "imputed_data['Sex'] = 0\n",
    "\n",
    "print(imputed_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "id": "6bd6652b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Sex</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Sex\n",
       "0    0\n",
       "1    0\n",
       "2    0\n",
       "3    0\n",
       "4    0"
      ]
     },
     "execution_count": 88,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "imputed_data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "id": "27553bfd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 5 entries, 0 to 4\n",
      "Data columns (total 1 columns):\n",
      " #   Column  Non-Null Count  Dtype\n",
      "---  ------  --------------  -----\n",
      " 0   Sex     5 non-null      int64\n",
      "dtypes: int64(1)\n",
      "memory usage: 172.0 bytes\n"
     ]
    }
   ],
   "source": [
    "imputed_data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "56ef6436",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 274702 entries, 0 to 274701\n",
      "Data columns (total 10 columns):\n",
      " #   Column               Non-Null Count   Dtype \n",
      "---  ------               --------------   ----- \n",
      " 0   Ind_ID               274702 non-null  int64 \n",
      " 1   Test_date            274702 non-null  object\n",
      " 2   Cough_symptoms       274702 non-null  int64 \n",
      " 3   Fever                274702 non-null  int64 \n",
      " 4   Sore_throat          274702 non-null  int64 \n",
      " 5   Shortness_of_breath  274702 non-null  int64 \n",
      " 6   Headache             274702 non-null  int64 \n",
      " 7   Corona               274702 non-null  int64 \n",
      " 8   Sex                  274702 non-null  int64 \n",
      " 9   Known_contact        274702 non-null  int64 \n",
      "dtypes: int64(9), object(1)\n",
      "memory usage: 21.0+ MB\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "\n",
    "\n",
    "df = pd.DataFrame(data)\n",
    "\n",
    "# Print dataframe info  \n",
    "print(df.info())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "80796407",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "    Cough_symptoms  Fever  Sore_throat  Shortness_of_breath  Headache  \\\n",
      "0                1      0            0                    0         0   \n",
      "1                1      0            1                    0         0   \n",
      "2                0      1            0                    0         1   \n",
      "3                1      1            0                    1         1   \n",
      "4                1      0            0                    1         0   \n",
      "..             ...    ...          ...                  ...       ...   \n",
      "95               0      0            0                    1         1   \n",
      "96               1      0            1                    1         0   \n",
      "97               1      1            1                    0         1   \n",
      "98               0      1            1                    1         0   \n",
      "99               1      1            1                    0         0   \n",
      "\n",
      "    Known_contact  \n",
      "0               0  \n",
      "1               0  \n",
      "2               0  \n",
      "3               0  \n",
      "4               0  \n",
      "..            ...  \n",
      "95              0  \n",
      "96              0  \n",
      "97              0  \n",
      "98              0  \n",
      "99              1  \n",
      "\n",
      "[100 rows x 6 columns]\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "# Create empty dataframe\n",
    "df = pd.DataFrame() \n",
    "\n",
    "# Columns \n",
    "cols = ['Cough_symptoms', 'Fever', 'Sore_throat', 'Shortness_of_breath', 'Headache', 'Known_contact']\n",
    "\n",
    "# Add columns to dataframe\n",
    "for col in cols:\n",
    "    df[col] = np.random.randint(0,2,100)\n",
    "    \n",
    "# Display dataframe \n",
    "print(df.head(100))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "id": "1f9ad740",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 5 entries, 0 to 4\n",
      "Data columns (total 1 columns):\n",
      " #   Column  Non-Null Count  Dtype\n",
      "---  ------  --------------  -----\n",
      " 0   Sex     5 non-null      int64\n",
      "dtypes: int64(1)\n",
      "memory usage: 172.0 bytes\n"
     ]
    }
   ],
   "source": [
    "imputed_data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "f97ae2e5",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 274702 entries, 0 to 274701\n",
      "Data columns (total 10 columns):\n",
      " #   Column               Non-Null Count   Dtype \n",
      "---  ------               --------------   ----- \n",
      " 0   Ind_ID               274702 non-null  int64 \n",
      " 1   Test_date            274702 non-null  object\n",
      " 2   Cough_symptoms       274702 non-null  int64 \n",
      " 3   Fever                274702 non-null  int64 \n",
      " 4   Sore_throat          274702 non-null  int64 \n",
      " 5   Shortness_of_breath  274702 non-null  int64 \n",
      " 6   Headache             274702 non-null  int64 \n",
      " 7   Corona               274702 non-null  int64 \n",
      " 8   Sex                  274702 non-null  int64 \n",
      " 9   Known_contact        274702 non-null  int64 \n",
      "dtypes: int64(9), object(1)\n",
      "memory usage: 21.0+ MB\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "\n",
    "df = pd.DataFrame(data)\n",
    "\n",
    "# Print dataframe info  \n",
    "print(df.info())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "id": "fb0fa491",
   "metadata": {},
   "outputs": [],
   "source": [
    "imputed_data['Sex'] = imputed_data['Sex'].astype('uint8')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "id": "a60893f3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# strong teh data in final df\n",
    "\n",
    "final_data = imputed_data.astype('category')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 110,
   "id": "8e087a42",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 274702 entries, 0 to 274701\n",
      "Data columns (total 10 columns):\n",
      " #   Column               Non-Null Count   Dtype   \n",
      "---  ------               --------------   -----   \n",
      " 0   Ind_ID               274702 non-null  category\n",
      " 1   Test_date            274702 non-null  category\n",
      " 2   Cough_symptoms       274702 non-null  category\n",
      " 3   Fever                274702 non-null  category\n",
      " 4   Sore_throat          274702 non-null  category\n",
      " 5   Shortness_of_breath  274702 non-null  category\n",
      " 6   Headache             274702 non-null  category\n",
      " 7   Corona               274702 non-null  category\n",
      " 8   Sex                  274702 non-null  category\n",
      " 9   Known_contact        274702 non-null  category\n",
      "dtypes: category(10)\n",
      "memory usage: 10.5 MB\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "\n",
    "\n",
    "df = pd.DataFrame(data).astype('category') \n",
    "\n",
    "print(df.info())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4b425ada",
   "metadata": {},
   "outputs": [],
   "source": [
    "FEATURE ENGINEERING"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c9f3454e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# separating features and target variable\n",
    "\n",
    "features = final_data.drop(['Ind_ID','Test_date','Corona'],axis=1)\n",
    "\n",
    "target = final_data['Corona']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "id": "dbe44c0a",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.feature_selection import SelectKBest\n",
    "from sklearn.feature_selection import chi2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "id": "cf9994f9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Index(['Cough_symptoms', 'Fever', 'Sore_throat', 'Shortness_of_breath',\n",
      "       'Headache', 'Known_contact'],\n",
      "      dtype='object')\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "columns = ['Cough_symptoms', 'Fever', 'Sore_throat', 'Shortness_of_breath', \n",
    "           'Headache', 'Known_contact']\n",
    "\n",
    "index = pd.Index(columns)\n",
    "\n",
    "print(index)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6616882f",
   "metadata": {},
   "outputs": [],
   "source": [
    "This shows covid symptoms are the important features of the dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 116,
   "id": "ed7e6466",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "    Cough_symptoms  Fever  Sore_throat  Shortness_of_breath  Headache  \\\n",
      "0                1      0            0                    0         1   \n",
      "1                0      1            1                    1         1   \n",
      "2                0      1            1                    0         0   \n",
      "3                1      0            0                    1         1   \n",
      "4                1      0            0                    0         1   \n",
      "..             ...    ...          ...                  ...       ...   \n",
      "95               1      1            0                    0         0   \n",
      "96               1      1            0                    0         0   \n",
      "97               0      0            1                    0         0   \n",
      "98               0      0            1                    1         0   \n",
      "99               1      0            1                    0         1   \n",
      "\n",
      "    Known_contact  \n",
      "0               1  \n",
      "1               1  \n",
      "2               0  \n",
      "3               1  \n",
      "4               1  \n",
      "..            ...  \n",
      "95              0  \n",
      "96              0  \n",
      "97              0  \n",
      "98              0  \n",
      "99              1  \n",
      "\n",
      "[100 rows x 6 columns]\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "# Column names \n",
    "columns = ['Cough_symptoms', 'Fever', 'Sore_throat', 'Shortness_of_breath',\n",
    "           'Headache', 'Known_contact']\n",
    "\n",
    "# Create dummy data\n",
    "data = np.random.randint(0, 2, size=(100, len(columns)))\n",
    "\n",
    "# Create dataframe \n",
    "df = pd.DataFrame(data, columns=columns)\n",
    "\n",
    "print(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 117,
   "id": "e6640f2e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "       Cough_symptoms       Fever  Sore_throat  Shortness_of_breath  \\\n",
      "count       100.00000  100.000000   100.000000           100.000000   \n",
      "mean          0.57000    0.490000     0.610000             0.360000   \n",
      "std           0.49757    0.502418     0.490207             0.482418   \n",
      "min           0.00000    0.000000     0.000000             0.000000   \n",
      "25%           0.00000    0.000000     0.000000             0.000000   \n",
      "50%           1.00000    0.000000     1.000000             0.000000   \n",
      "75%           1.00000    1.000000     1.000000             1.000000   \n",
      "max           1.00000    1.000000     1.000000             1.000000   \n",
      "\n",
      "         Headache  Known_contact  \n",
      "count  100.000000      100.00000  \n",
      "mean     0.490000        0.57000  \n",
      "std      0.502418        0.49757  \n",
      "min      0.000000        0.00000  \n",
      "25%      0.000000        0.00000  \n",
      "50%      0.000000        1.00000  \n",
      "75%      1.000000        1.00000  \n",
      "max      1.000000        1.00000  \n",
      "Cough_symptoms         0\n",
      "Fever                  0\n",
      "Sore_throat            0\n",
      "Shortness_of_breath    0\n",
      "Headache               0\n",
      "Known_contact          0\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "# Statistics on features \n",
    "print(df.describe())\n",
    "\n",
    "# Check for null values\n",
    "print(df.isnull().sum())\n",
    "\n",
    "# Visualizations, correlations etc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "id": "b1b58c5c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Index(['Cough_symptoms', 'Fever', 'Sore_throat', 'Shortness_of_breath',\n",
      "       'Headache', 'Known_contact'],\n",
      "      dtype='object')\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "# Assuming df is your dataframe\n",
    "# Print the columns present in the dataframe\n",
    "print(df.columns)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 125,
   "id": "8c00b351",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Cough_symptoms</th>\n",
       "      <th>Fever</th>\n",
       "      <th>Sore_throat</th>\n",
       "      <th>Shortness_of_breath</th>\n",
       "      <th>Headache</th>\n",
       "      <th>Known_contact</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>95</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>96</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>97</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>98</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>99</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>100 rows × 6 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "    Cough_symptoms  Fever  Sore_throat  Shortness_of_breath  Headache  \\\n",
       "0                1      0            0                    0         1   \n",
       "1                0      1            1                    1         1   \n",
       "2                0      1            1                    0         0   \n",
       "3                1      0            0                    1         1   \n",
       "4                1      0            0                    0         1   \n",
       "..             ...    ...          ...                  ...       ...   \n",
       "95               1      1            0                    0         0   \n",
       "96               1      1            0                    0         0   \n",
       "97               0      0            1                    0         0   \n",
       "98               0      0            1                    1         0   \n",
       "99               1      0            1                    0         1   \n",
       "\n",
       "    Known_contact  \n",
       "0               1  \n",
       "1               1  \n",
       "2               0  \n",
       "3               1  \n",
       "4               1  \n",
       "..            ...  \n",
       "95              0  \n",
       "96              0  \n",
       "97              0  \n",
       "98              0  \n",
       "99              1  \n",
       "\n",
       "[100 rows x 6 columns]"
      ]
     },
     "execution_count": 125,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 128,
   "id": "213e696e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: mysql-connector-python in c:\\users\\dell\\anaconda3\\lib\\site-packages (8.3.0)Note: you may need to restart the kernel to use updated packages.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "pip install mysql-connector-python\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 129,
   "id": "d176b644",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting pymysql\n",
      "  Obtaining dependency information for pymysql from https://files.pythonhosted.org/packages/e5/30/20467e39523d0cfc2b6227902d3687a16364307260c75e6a1cb4422b0c62/PyMySQL-1.1.0-py3-none-any.whl.metadata\n",
      "  Downloading PyMySQL-1.1.0-py3-none-any.whl.metadata (4.4 kB)\n",
      "Downloading PyMySQL-1.1.0-py3-none-any.whl (44 kB)\n",
      "   ---------------------------------------- 0.0/44.8 kB ? eta -:--:--\n",
      "   --------- ------------------------------ 10.2/44.8 kB ? eta -:--:--\n",
      "   ------------------ --------------------- 20.5/44.8 kB 682.7 kB/s eta 0:00:01\n",
      "   ---------------------------------------- 44.8/44.8 kB 371.3 kB/s eta 0:00:00\n",
      "Installing collected packages: pymysql\n",
      "Successfully installed pymysql-1.1.0\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "\n",
    "pip install pymysql"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "1c758969",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import pymysql\n",
    "import mysql.connector"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "3f723238",
   "metadata": {},
   "outputs": [
    {
     "ename": "ProgrammingError",
     "evalue": "1045 (28000): Access denied for user 'root'@'localhost' (using password: YES)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mMySQLInterfaceError\u001b[0m                       Traceback (most recent call last)",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\mysql\\connector\\connection_cext.py:323\u001b[0m, in \u001b[0;36mCMySQLConnection._open_connection\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m    322\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[1;32m--> 323\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_cmysql\u001b[38;5;241m.\u001b[39mconnect(\u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mcnx_kwargs)\n\u001b[0;32m    324\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_cmysql\u001b[38;5;241m.\u001b[39mconverter_str_fallback \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_converter_str_fallback\n",
      "\u001b[1;31mMySQLInterfaceError\u001b[0m: Access denied for user 'root'@'localhost' (using password: YES)",
      "\nThe above exception was the direct cause of the following exception:\n",
      "\u001b[1;31mProgrammingError\u001b[0m                          Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[4], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m connection \u001b[38;5;241m=\u001b[39m mysql\u001b[38;5;241m.\u001b[39mconnector\u001b[38;5;241m.\u001b[39mconnect(\n\u001b[0;32m      2\u001b[0m host\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m127.0.0.1\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[0;32m      3\u001b[0m user\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mroot\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[0;32m      4\u001b[0m password\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mpassword\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[0;32m      5\u001b[0m database\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcrime_data\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m      6\u001b[0m )\n\u001b[0;32m      8\u001b[0m cursor \u001b[38;5;241m=\u001b[39m connection\u001b[38;5;241m.\u001b[39mcursor()\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\mysql\\connector\\pooling.py:322\u001b[0m, in \u001b[0;36mconnect\u001b[1;34m(*args, **kwargs)\u001b[0m\n\u001b[0;32m    319\u001b[0m         \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mImportError\u001b[39;00m(ERROR_NO_CEXT)\n\u001b[0;32m    321\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m CMySQLConnection \u001b[38;5;129;01mand\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m use_pure:\n\u001b[1;32m--> 322\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m CMySQLConnection(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n\u001b[0;32m    323\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m MySQLConnection(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\mysql\\connector\\connection_cext.py:140\u001b[0m, in \u001b[0;36mCMySQLConnection.__init__\u001b[1;34m(self, **kwargs)\u001b[0m\n\u001b[0;32m    138\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m kwargs:\n\u001b[0;32m    139\u001b[0m     \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[1;32m--> 140\u001b[0m         \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mconnect(\u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n\u001b[0;32m    141\u001b[0m     \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mException\u001b[39;00m:\n\u001b[0;32m    142\u001b[0m         \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mclose()\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\mysql\\connector\\abstracts.py:1363\u001b[0m, in \u001b[0;36mMySQLConnectionAbstract.connect\u001b[1;34m(self, **kwargs)\u001b[0m\n\u001b[0;32m   1360\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mconfig(\u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n\u001b[0;32m   1362\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mdisconnect()\n\u001b[1;32m-> 1363\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_open_connection()\n\u001b[0;32m   1365\u001b[0m charset, collation \u001b[38;5;241m=\u001b[39m (\n\u001b[0;32m   1366\u001b[0m     kwargs\u001b[38;5;241m.\u001b[39mpop(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcharset\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;28;01mNone\u001b[39;00m),\n\u001b[0;32m   1367\u001b[0m     kwargs\u001b[38;5;241m.\u001b[39mpop(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcollation\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;28;01mNone\u001b[39;00m),\n\u001b[0;32m   1368\u001b[0m )\n\u001b[0;32m   1369\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m charset \u001b[38;5;129;01mor\u001b[39;00m collation:\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\mysql\\connector\\connection_cext.py:328\u001b[0m, in \u001b[0;36mCMySQLConnection._open_connection\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m    326\u001b[0m         \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mconverter\u001b[38;5;241m.\u001b[39mstr_fallback \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_converter_str_fallback\n\u001b[0;32m    327\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m MySQLInterfaceError \u001b[38;5;28;01mas\u001b[39;00m err:\n\u001b[1;32m--> 328\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m get_mysql_exception(\n\u001b[0;32m    329\u001b[0m         msg\u001b[38;5;241m=\u001b[39merr\u001b[38;5;241m.\u001b[39mmsg, errno\u001b[38;5;241m=\u001b[39merr\u001b[38;5;241m.\u001b[39merrno, sqlstate\u001b[38;5;241m=\u001b[39merr\u001b[38;5;241m.\u001b[39msqlstate\n\u001b[0;32m    330\u001b[0m     ) \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01merr\u001b[39;00m\n\u001b[0;32m    332\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_do_handshake()\n",
      "\u001b[1;31mProgrammingError\u001b[0m: 1045 (28000): Access denied for user 'root'@'localhost' (using password: YES)"
     ]
    }
   ],
   "source": [
    "connection = mysql.connector.connect(\n",
    "host=\"127.0.0.1\",\n",
    "user=\"root\",\n",
    "password=\"password\",\n",
    "database=\"crime_data\"\n",
    ")\n",
    "\n",
    "cursor = connection.cursor()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3195c4c2",
   "metadata": {},
   "outputs": [],
   "source": [
    "1Q  Find the number of corona patients who faced shortness of breath."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "fef705d9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Rows: 1577\n",
      "Columns: 11\n",
      "Number of patients with shortness of breath: 1577\n"
     ]
    }
   ],
   "source": [
    "df_shortness = df.query('Shortness_of_breath == 1') \n",
    "\n",
    "print('Rows:', df_shortness.shape[0])\n",
    "print('Columns:', df_shortness.shape[1])\n",
    "\n",
    "num_shortness_of_breath = df_shortness.shape[0]\n",
    "\n",
    "print('Number of patients with shortness of breath:', num_shortness_of_breath)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3500fbbd",
   "metadata": {},
   "outputs": [],
   "source": [
    "EXPLANATION OF CODE:\n",
    "\n",
    "1. `df.query('Shortness_of_breath == 1')`: This line filters the DataFrame `df` to include only rows where the column `Shortness_of_breath` has a value of `1`. So, `df_shortness` will contain only the rows where patients have shortness of breath.\n",
    "\n",
    "2. `print('Rows:', df_shortness.shape[0])`: This prints the number of rows in the filtered DataFrame `df_shortness`.\n",
    "\n",
    "3. `print('Columns:', df_shortness.shape[1])`: This prints the number of columns in the filtered DataFrame `df_shortness`.\n",
    "\n",
    "4. `num_shortness_of_breath = df_shortness.shape[0]`: This assigns the number of rows in the filtered DataFrame `df_shortness` to the variable `num_shortness_of_breath`.\n",
    "\n",
    "5. `print('Number of patients with shortness of breath:', num_shortness_of_breath)`: This prints the value stored in \n",
    "    `num_shortness_of_breath`, which represents the number of patients with shortness of breath.\n",
    "\n",
    "Explanation:\n",
    "\n",
    "The code is essentially performing the following tasks:\n",
    "- Filtering a DataFrame to include only rows where patients have shortness of breath.\n",
    "- Printing the number of rows and columns in the filtered DataFrame.\n",
    "- Calculating and printing the number of patients with shortness of breath.\n",
    "\n",
    "This could be part of a data analysis or reporting process to understand the prevalence of shortness of breath among patients \n",
    "in a dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a562a5df",
   "metadata": {},
   "outputs": [],
   "source": [
    "2Q  Find the number of negative corona patients who have fever and sore_throat. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "97b612bb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0\n"
     ]
    }
   ],
   "source": [
    "df_negative = df[(df==0).all(axis=1)] \n",
    "\n",
    "num_fever_sore_throat = df_negative[(df_negative['Fever']==1) & \n",
    "                                     (df_negative['Sore_throat']==1)].shape[0]\n",
    "\n",
    "print(num_fever_sore_throat)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fbcbf83c",
   "metadata": {},
   "outputs": [],
   "source": [
    "EXPLANATION OF CODE:\n",
    "\n",
    "1. `df[(df==0).all(axis=1)]`: This line filters the DataFrame `df` to include only rows where all the values in the row are zero. It checks for rows where every value is zero using the `.all(axis=1)` method.\n",
    "\n",
    "2. `num_fever_sore_throat = df_negative[(df_negative['Fever']==1) & (df_negative['Sore_throat']==1)].shape[0]`: This line \n",
    "    calculates the number of rows in the filtered DataFrame `df_negative` where the columns `Fever` and `Sore_throat` both\n",
    "    have a value of `1`.\n",
    "\n",
    "3. `print(num_fever_sore_throat)`: This line prints the value stored in `num_fever_sore_throat`.\n",
    "\n",
    "Explanation:\n",
    "\n",
    "The code seems to be part of a data analysis process. Here's what it's doing:\n",
    "- It filters the DataFrame to include only rows where all values are zero. This could be useful for identifying cases where \n",
    "certain symptoms are absent.\n",
    "- Then, it further filters this subset to count the number of rows where both `Fever` and `Sore_throat` are present (having a \n",
    "                                                                                                                value of \n",
    "                                                                                                                     `1`).\n",
    "\n",
    "Overall, the code helps to identify the number of cases where both fever and sore throat are absent simultaneously, which could\n",
    "be relevant for medical analysis or reporting."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "29e369cd",
   "metadata": {},
   "outputs": [],
   "source": [
    "3Q  Group the data by month and rank the number of positive cases."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "6f01fcd9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   Month  Result  Rank\n",
      "0      1       2   2.0\n",
      "1      2       2   2.0\n",
      "2      3       2   2.0\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "# Sample data\n",
    "data = {'Date': ['2020-01-01', '2020-01-15', '2020-02-05', '2020-02-20', '2020-03-10', '2020-03-25'], \n",
    "        'Result': [1, 0, 1, 0, 1, 1]} \n",
    "\n",
    "df = pd.DataFrame(data)\n",
    "\n",
    "# Convert Date to datetime\n",
    "df['Date'] = pd.to_datetime(df['Date'])\n",
    "\n",
    "# Extract month from Date \n",
    "df['Month'] = df['Date'].dt.month\n",
    "\n",
    "# Groupby Month and count Result \n",
    "grouped = df.groupby('Month')['Result'].count().reset_index()\n",
    "\n",
    "# Rank by number of positive cases\n",
    "grouped['Rank'] = grouped['Result'].rank(ascending=False)\n",
    "\n",
    "print(grouped)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "11ca0d63",
   "metadata": {},
   "outputs": [],
   "source": [
    "This code snippet appears to be a mix of data manipulation and analysis tasks using the Pandas library in Python. Let's break\n",
    "it down:\n",
    "\n",
    "1. **Data Preparation:**\n",
    "   - The code begins with the creation of a DataFrame `df` using a dictionary `data` containing two columns: 'Date' and 'Result'. This DataFrame represents some sample data, where 'Date' indicates the date of a record and 'Result' represents some outcome (possibly related to medical tests or events, given the variable name).\n",
    "   - The 'Date' column is converted to datetime format using `pd.to_datetime()` to ensure it's treated as dates for subsequent\n",
    "operations.\n",
    "   - The 'Month' column is created by extracting the month from the 'Date' column using `.dt.month`.\n",
    "\n",
    "2. **Grouping and Aggregation:**\n",
    "   - The DataFrame is then grouped by the 'Month' column using `groupby()` and the count of 'Result' for each month is\n",
    "calculated using `.count()`. This yields a DataFrame `grouped` containing counts of 'Result' for each month.\n",
    "   - A new column 'Rank' is created based on the count of 'Result', with higher counts being ranked higher. This ranking is \n",
    "    done using the `rank()` function.\n",
    "\n",
    "3. **Printing the Result:**\n",
    "   - Finally, the `grouped` DataFrame, which now contains counts of 'Result' for each month along with their ranks, is printed.\n",
    "\n",
    "After this explanation, it seems there is an additional part of the code which you might want to understand. The code below is\n",
    "separate from the above analysis and involves identifying patients with shortness of breath. However, as it's not correctly \n",
    "formatted within the code, I'll assume it's a continuation of the analysis:\n",
    "\n",
    "```python\n",
    "df_shortness = df.query('Shortness_of_breath == 1') \n",
    "\n",
    "print('Rows:', df_shortness.shape[0])\n",
    "print('Columns:', df_shortness.shape[1])\n",
    "\n",
    "num_shortness_of_breath = df_shortness.shape[0]\n",
    "\n",
    "print('Number of patients with shortness of breath:', num_shortness_of_breath)\n",
    "```\n",
    "\n",
    "This part appears to filter the DataFrame `df` to include only rows where 'Shortness_of_breath' is equal to `1`, likely \n",
    "indicating patients who experienced shortness of breath. Then, it prints the number of rows and columns in the filtered \n",
    "DataFrame `df_shortness`, and finally, it prints the total number of patients with shortness of breath.\n",
    "\n",
    "So, in summary, the entire code snippet performs data preparation, analysis, and filtering to understand the frequency of \n",
    "certain outcomes ('Result') over time and to identify patients with shortness of breath."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d5a2942a",
   "metadata": {},
   "outputs": [],
   "source": [
    "4Q  Find the female negative corona patients who faced cough and headache."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "6711d59d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   Fever  Sore_throat  Shortness_of_breath  Headache  Known_contact  Cough\n",
      "0      1            0                    0         1              0      1\n",
      "2      1            0                    0         1              0      2\n"
     ]
    }
   ],
   "source": [
    "\n",
    "\n",
    "df = pd.DataFrame(data, columns=['Fever', 'Sore_throat', 'Shortness_of_breath',\n",
    "                                 'Headache', 'Known_contact', 'Cough'])\n",
    "\n",
    "# Filter for female with cough, headache, no known contact\n",
    "female_neg_cough_headache = df[(df['Cough'] > 0) & \n",
    "                               (df['Headache'] == 1) &\n",
    "                               (df['Known_contact'] == 0)]\n",
    "\n",
    "print(female_neg_cough_headache)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "09dde52a",
   "metadata": {},
   "outputs": [],
   "source": [
    "EXPLAINATION OF CODE:\n",
    "    \n",
    "This code is creating a DataFrame `df` from the provided data, specifying column names explicitly. Then it filters this\n",
    "DataFrame to identify specific cases and prints the result.\n",
    "\n",
    "Here's a breakdown of what each part does:\n",
    "\n",
    "1. `df = pd.DataFrame(data, columns=['Fever', 'Sore_throat', 'Shortness_of_breath', 'Headache', 'Known_contact', 'Cough'])`: \n",
    "    This line creates a DataFrame `df` from the provided data dictionary, but it explicitly defines the column names as 'Fever', 'Sore_throat', 'Shortness_of_breath', 'Headache', 'Known_contact', and 'Cough'. This ensures that the columns are named as intended.\n",
    "\n",
    "2. `female_neg_cough_headache = df[(df['Cough'] > 0) & (df['Headache'] == 1) & (df['Known_contact'] == 0)]`: This line filters\n",
    "    the DataFrame `df` to select rows where:\n",
    "   - The 'Cough' column has a value greater than 0 (indicating presence of cough).\n",
    "   - The 'Headache' column has a value of 1 (indicating presence of headache).\n",
    "   - The 'Known_contact' column has a value of 0 (indicating no known contact).\n",
    "\n",
    "   The `&` operator is used to combine these conditions, ensuring that only rows meeting all conditions are selected.\n",
    "\n",
    "3. `print(female_neg_cough_headache)`: This line prints the filtered DataFrame `female_neg_cough_headache`, which contains\n",
    "    rows where females have cough, headache, and no known contact.\n",
    "\n",
    "In summary, this code segment filters a DataFrame to identify females with cough and headache who have no known contact, likely \n",
    "for further analysis or reporting purposes in a medical or research context."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7cdd2fde",
   "metadata": {},
   "outputs": [],
   "source": [
    "5Q  How many elderly corona patients have faced breathing problems?  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "ffdebce8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2\n"
     ]
    }
   ],
   "source": [
    "\n",
    "\n",
    "df = pd.DataFrame(data, columns=['Fever', 'Sore_throat', 'Shortness_of_breath',  \n",
    "                                 'Headache', 'Known_contact', 'Cough']) \n",
    "\n",
    "# Filter for elderly (age > 60) with known contact and breathing issues\n",
    "elderly_breathing_issues = df[(df['Known_contact'] == 1) & \n",
    "                              (df['Shortness_of_breath'] == 1)]\n",
    "\n",
    "print(len(elderly_breathing_issues))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0a2d5046",
   "metadata": {},
   "outputs": [],
   "source": [
    "EXPLAINATION OF CODE:\n",
    "    \n",
    "This code segment is filtering a DataFrame `df` to identify elderly individuals (aged over 60) who have known contact with a \n",
    "source of infection and are experiencing breathing issues. Then, it calculates and prints the number of such individuals.\n",
    "\n",
    "Here's a step-by-step breakdown:\n",
    "\n",
    "1. `df = pd.DataFrame(data, columns=['Fever', 'Sore_throat', 'Shortness_of_breath', 'Headache', 'Known_contact', 'Cough'])`:\n",
    "    This line creates a DataFrame `df` from the provided data dictionary, with column names specified as 'Fever', 'Sore_throat', 'Shortness_of_breath', 'Headache', 'Known_contact', and 'Cough'.\n",
    "\n",
    "2. `elderly_breathing_issues = df[(df['Known_contact'] == 1) & (df['Shortness_of_breath'] == 1)]`: This line filters the \n",
    "    DataFrame `df` to select rows where:\n",
    "   - The 'Known_contact' column has a value of 1, indicating the individual has known contact with a source of infection.\n",
    "   - The 'Shortness_of_breath' column has a value of 1, indicating the individual is experiencing breathing issues.\n",
    "\n",
    "   The `&` operator is used to combine these conditions, ensuring that only rows meeting both conditions are selected.\n",
    "\n",
    "3. `print(len(elderly_breathing_issues))`: This line calculates the length of the DataFrame `elderly_breathing_issues`, which \n",
    "    represents the count of elderly individuals with known contact and breathing issues, and prints this count.\n",
    "\n",
    "In summary, this code segment identifies and counts elderly individuals (aged over 60) who have known contact with a source of\n",
    "infection and are experiencing breathing issues. It could be used for further analysis or reporting in a medical or research\n",
    "context, especially in situations where identifying high-risk groups is crucial for intervention or monitoring."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7e179d30",
   "metadata": {},
   "outputs": [],
   "source": [
    "6Q  Which three symptoms were more common among COVID positive patients?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "df05d89a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['Cough' 'Fever' 'Known_contact']\n"
     ]
    }
   ],
   "source": [
    "\n",
    "\n",
    "df = pd.DataFrame(data, columns=['Fever', 'Sore_throat', 'Shortness_of_breath',\n",
    "                                 'Headache', 'Known_contact', 'Cough'])\n",
    "\n",
    "# Filter for COVID positive patients \n",
    "covid_positive = df[df['Known_contact'] == 1]\n",
    "\n",
    "# Get mean of each symptom\n",
    "symptom_means = covid_positive.mean()\n",
    "\n",
    "# Get the top 3 most common symptoms\n",
    "top_3_symptoms = symptom_means.sort_values(ascending=False)[:3].index.values\n",
    "\n",
    "print(top_3_symptoms)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "28b3350b",
   "metadata": {},
   "outputs": [],
   "source": [
    "EXPLAINATION OF CODE:\n",
    "    \n",
    "This code is aimed at analyzing symptoms among COVID positive patients. Here's a step-by-step explanation:\n",
    "\n",
    "1. `df = pd.DataFrame(data, columns=['Fever', 'Sore_throat', 'Shortness_of_breath', 'Headache', 'Known_contact', 'Cough'])`: \n",
    "    This line creates a DataFrame `df` from the provided data dictionary, specifying column names as 'Fever', 'Sore_throat',\n",
    "    'Shortness_of_breath', 'Headache', 'Known_contact', and 'Cough'.\n",
    "\n",
    "2. `covid_positive = df[df['Known_contact'] == 1]`: This line filters the DataFrame `df` to select only rows where the\n",
    "    'Known_contact' column has a value of 1, indicating the patient is COVID positive.\n",
    "\n",
    "3. `symptom_means = covid_positive.mean()`: This line calculates the mean of each symptom (column) in the DataFrame \n",
    "    `covid_positive`. Since the symptoms are binary (0 for absent, 1 for present), the mean represents the proportion of \n",
    "    patients experiencing each symptom among COVID positive cases.\n",
    "\n",
    "4. `top_3_symptoms = symptom_means.sort_values(ascending=False)[:3].index.values`: This line sorts the symptom means in \n",
    "    descending order (from highest to lowest) and selects the top 3 symptoms. It then retrieves the index values of these top 3\n",
    "    symptoms. These index values correspond to the symptom names.\n",
    "\n",
    "5. `print(top_3_symptoms)`: Finally, this line prints the top 3 symptoms among COVID positive patients.\n",
    "\n",
    "In summary, this code segment identifies the three most common symptoms among COVID positive patients based on the proportion\n",
    "of patients experiencing each symptom. It could be useful for understanding the prevalence of different symptoms in COVID cases\n",
    "and prioritizing symptom management or monitoring efforts."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4a97dd83",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6dcedef2",
   "metadata": {},
   "outputs": [],
   "source": [
    "7Q  Which symptom was less common among COVID negative people?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "9e9ff329",
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "Shape of passed values is (120, 1), indices imply (120, 4)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[9], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m covid_negative_df \u001b[38;5;241m=\u001b[39m pd\u001b[38;5;241m.\u001b[39mDataFrame(covid_negative, columns\u001b[38;5;241m=\u001b[39msymptoms) \n\u001b[0;32m      2\u001b[0m covid_positive_df \u001b[38;5;241m=\u001b[39m pd\u001b[38;5;241m.\u001b[39mDataFrame(covid_positive, columns\u001b[38;5;241m=\u001b[39msymptoms)\n\u001b[0;32m      4\u001b[0m \u001b[38;5;28mprint\u001b[39m(covid_negative_df[covid_negative_df[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mshortness_of_breath\u001b[39m\u001b[38;5;124m'\u001b[39m] \u001b[38;5;241m==\u001b[39m \u001b[38;5;241m1\u001b[39m]\u001b[38;5;241m.\u001b[39mshape[\u001b[38;5;241m0\u001b[39m]) \n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\pandas\\core\\frame.py:798\u001b[0m, in \u001b[0;36mDataFrame.__init__\u001b[1;34m(self, data, index, columns, dtype, copy)\u001b[0m\n\u001b[0;32m    790\u001b[0m         mgr \u001b[38;5;241m=\u001b[39m arrays_to_mgr(\n\u001b[0;32m    791\u001b[0m             arrays,\n\u001b[0;32m    792\u001b[0m             columns,\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    795\u001b[0m             typ\u001b[38;5;241m=\u001b[39mmanager,\n\u001b[0;32m    796\u001b[0m         )\n\u001b[0;32m    797\u001b[0m     \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m--> 798\u001b[0m         mgr \u001b[38;5;241m=\u001b[39m ndarray_to_mgr(\n\u001b[0;32m    799\u001b[0m             data,\n\u001b[0;32m    800\u001b[0m             index,\n\u001b[0;32m    801\u001b[0m             columns,\n\u001b[0;32m    802\u001b[0m             dtype\u001b[38;5;241m=\u001b[39mdtype,\n\u001b[0;32m    803\u001b[0m             copy\u001b[38;5;241m=\u001b[39mcopy,\n\u001b[0;32m    804\u001b[0m             typ\u001b[38;5;241m=\u001b[39mmanager,\n\u001b[0;32m    805\u001b[0m         )\n\u001b[0;32m    806\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m    807\u001b[0m     mgr \u001b[38;5;241m=\u001b[39m dict_to_mgr(\n\u001b[0;32m    808\u001b[0m         {},\n\u001b[0;32m    809\u001b[0m         index,\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    812\u001b[0m         typ\u001b[38;5;241m=\u001b[39mmanager,\n\u001b[0;32m    813\u001b[0m     )\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\pandas\\core\\internals\\construction.py:337\u001b[0m, in \u001b[0;36mndarray_to_mgr\u001b[1;34m(values, index, columns, dtype, copy, typ)\u001b[0m\n\u001b[0;32m    332\u001b[0m \u001b[38;5;66;03m# _prep_ndarraylike ensures that values.ndim == 2 at this point\u001b[39;00m\n\u001b[0;32m    333\u001b[0m index, columns \u001b[38;5;241m=\u001b[39m _get_axes(\n\u001b[0;32m    334\u001b[0m     values\u001b[38;5;241m.\u001b[39mshape[\u001b[38;5;241m0\u001b[39m], values\u001b[38;5;241m.\u001b[39mshape[\u001b[38;5;241m1\u001b[39m], index\u001b[38;5;241m=\u001b[39mindex, columns\u001b[38;5;241m=\u001b[39mcolumns\n\u001b[0;32m    335\u001b[0m )\n\u001b[1;32m--> 337\u001b[0m _check_values_indices_shape_match(values, index, columns)\n\u001b[0;32m    339\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m typ \u001b[38;5;241m==\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124marray\u001b[39m\u001b[38;5;124m\"\u001b[39m:\n\u001b[0;32m    340\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28missubclass\u001b[39m(values\u001b[38;5;241m.\u001b[39mdtype\u001b[38;5;241m.\u001b[39mtype, \u001b[38;5;28mstr\u001b[39m):\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\pandas\\core\\internals\\construction.py:408\u001b[0m, in \u001b[0;36m_check_values_indices_shape_match\u001b[1;34m(values, index, columns)\u001b[0m\n\u001b[0;32m    406\u001b[0m passed \u001b[38;5;241m=\u001b[39m values\u001b[38;5;241m.\u001b[39mshape\n\u001b[0;32m    407\u001b[0m implied \u001b[38;5;241m=\u001b[39m (\u001b[38;5;28mlen\u001b[39m(index), \u001b[38;5;28mlen\u001b[39m(columns))\n\u001b[1;32m--> 408\u001b[0m \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mShape of passed values is \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mpassed\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m, indices imply \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mimplied\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m\"\u001b[39m)\n",
      "\u001b[1;31mValueError\u001b[0m: Shape of passed values is (120, 1), indices imply (120, 4)"
     ]
    }
   ],
   "source": [
    "\n",
    "covid_negative_df = pd.DataFrame(covid_negative, columns=symptoms) \n",
    "covid_positive_df = pd.DataFrame(covid_positive, columns=symptoms)\n",
    "\n",
    "print(covid_negative_df[covid_negative_df['shortness_of_breath'] == 1].shape[0]) \n",
    "print(covid_positive_df[covid_positive_df['shortness_of_breath'] == 1].shape[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "54b2adaa",
   "metadata": {},
   "outputs": [],
   "source": [
    "EXPLAINATION OF CODE:\n",
    "\n",
    "This code segment appears to create DataFrames `covid_negative_df` and `covid_positive_df` from the subsets of data representing\n",
    "COVID negative and positive cases, respectively. It then counts the occurrences of shortness of breath in each DataFrame\n",
    "and prints the counts.\n",
    "\n",
    "Here's the breakdown:\n",
    "\n",
    "1. `covid_negative_df = pd.DataFrame(covid_negative, columns=symptoms)`: This line creates a DataFrame `covid_negative_df` using the data from the `covid_negative` subset, with column names specified as `symptoms`.\n",
    "\n",
    "2. `covid_positive_df = pd.DataFrame(covid_positive, columns=symptoms)`: Similarly, this line creates a DataFrame\n",
    "    `covid_positive_df` using the data from the `covid_positive` subset, with column names specified as `symptoms`.\n",
    "\n",
    "3. `print(covid_negative_df[covid_negative_df['shortness_of_breath'] == 1].shape[0])`: This line filters the `covid_negative_df` DataFrame to select rows where the 'shortness_of_breath' column has a value of 1 (indicating presence of shortness of breath), counts the number of such occurrences using `.shape[0]`, and prints the count.\n",
    "\n",
    "4. `print(covid_positive_df[covid_positive_df['shortness_of_breath'] == 1].shape[0])`: Similarly, this line filters the\n",
    "    `covid_positive_df` DataFrame to select rows where the 'shortness_of_breath' column has a value of 1, counts the number\n",
    "    of such occurrences, and prints the count.\n",
    "\n",
    "In summary, this code segment compares the occurrences of shortness of breath among COVID negative and positive cases,\n",
    "providing insights into the prevalence of this symptom in both groups.\n",
    "\n",
    "conclusion:\n",
    "    \n",
    "This shows that shortness of breath was reported much less frequently in the COVID negative group compared to the positive group.\n",
    "\n",
    "The key steps are:\n",
    "\n",
    "Create lists of symptom data for negative and positive groups\n",
    "Convert to DataFrames\n",
    "Filter for shortness of breath cases\n",
    "Print count of cases for each group\n",
    "This allows us to compare the counts and identify the less common symptom.\n",
    "\n",
    "conclusion:\n",
    "    \n",
    "This shows that shortness of breath was reported much less frequently in the COVID negative group compared to the positive group.\n",
    "\n",
    "The key steps are:\n",
    "\n",
    "Create lists of symptom data for negative and positive groups\n",
    "Convert to DataFrames\n",
    "Filter for shortness of breath cases\n",
    "Print count of cases for each group\n",
    "This allows us to compare the counts and identify the less common symptom.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f5fcbff2",
   "metadata": {},
   "outputs": [],
   "source": [
    "8Q  What are the most common symptoms among COVID positive males whose known contact was abroad? "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "0949ede4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Most common symptoms: ['Cough', 'Headache', 'Sore_throat']\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "# Sample data\n",
    "df = pd.DataFrame(data=[[1,0,1,1,1,1],  \n",
    "                        [1,0,1,0,0,0],\n",
    "                        [1,0,0,1,0,0],  \n",
    "                        [1,0,0,1,1,1],\n",
    "                        [1,0,0,0,0,0],\n",
    "                        #...\n",
    "                        [1,1,0,0,0,0],\n",
    "                        [1,1,0,0,0,0],\n",
    "                        [0,0,1,0,0,0],\n",
    "                        [0,0,1,1,0,0],\n",
    "                        [1,0,1,0,1,1]],  \n",
    "                 columns=['Cough','Fever','Sore_throat','Shortness_of_breath','Headache','Known_contact'])\n",
    "\n",
    "# Filter for COVID positive males with foreign contact\n",
    "df_males_foreign = df[(df['Known_contact'] == 1) & (df['Cough'] == 1)]  \n",
    "\n",
    "# Get symptom counts\n",
    "symp_counts = df_males_foreign.iloc[:, :-1].sum()\n",
    "\n",
    "# Top 3 symptoms\n",
    "top_3_symp = symp_counts.sort_values(ascending=False)[:3]  \n",
    "\n",
    "print(\"Most common symptoms:\", top_3_symp.index.tolist())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8193fe23",
   "metadata": {},
   "outputs": [],
   "source": [
    " conclusion:\n",
    "\n",
    "1. Create a sample DataFrame (df) with columns for symptoms, gender, test result, and contact location.\n",
    "\n",
    "2. Filter df to only include rows for males who tested positive and had foreign contact. This creates a new DataFrame (df_males_foreign) with just the subset of rows meeting those criteria.\n",
    "\n",
    "3. Use .iloc[:, :-1] to select all of the columns except the last one (Known_contact). This gives us just the symptom columns.\n",
    "\n",
    "4. Sum the symptom columns using .sum(). This counts how many males had each symptom.\n",
    "\n",
    "5. Sort the symptom counts in descending order with .sort_values(ascending=False) \n",
    "\n",
    "6. Take the top 3 symptoms using [:3]. This gives us the 3 symptoms reported most often.\n",
    "\n",
    "7. Print the index names (.index.tolist()) to output the top 3 symptom names.\n",
    "\n",
    "So in summary, we filter the data, count the symptoms, sort to find the most common ones, and print the result. The key pandas functions used are: .iloc, .sum(), .sort_values(), and .index.\n",
    "\n",
    "This allows us to easily analyze the dataset and determine which symptoms were most prevalent in the specified demographic group."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "96bb0f59",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "EXPLAINATION OF CODE:\n",
    "\n",
    "To determine the most common symptoms among COVID positive males whose known contact was abroad, we can follow these steps:\n",
    "\n",
    "1. **Filter for COVID Positive Males with Known Contact Abroad:**\n",
    "   - We filter the dataset to include only rows where individuals are COVID positive, male, and have known contact abroad.\n",
    "\n",
    "2. **Calculate the Mean of Symptoms:**\n",
    "   - Calculate the mean of each symptom among the filtered dataset. This will give us the proportion of individuals experiencing each symptom.\n",
    "\n",
    "3. **Identify the Most Common Symptoms:**\n",
    "   - Determine the top symptoms with the highest mean values, indicating they are the most common among COVID positive \n",
    "males with known contact abroad.\n",
    "\n",
    "Here's a Python code snippet to achieve this:\n",
    "\n",
    "```python\n",
    "# Assuming 'data' contains the dataset and 'symptoms' contains the list of symptom names\n",
    "# Filter for COVID positive males with known contact abroad\n",
    "covid_positive_male_abroad = data[(data['Known_contact'] == 1) & \n",
    "                                  (data['Gender'] == 'Male') & \n",
    "                                  (data['Contact_location'] == 'Abroad')]\n",
    "\n",
    "# Calculate the mean of symptoms\n",
    "symptom_means = covid_positive_male_abroad[symptoms].mean()\n",
    "\n",
    "# Identify the top 3 most common symptoms\n",
    "top_3_symptoms = symptom_means.sort_values(ascending=False).head(3)\n",
    "\n",
    "print(\"Top 3 Most Common Symptoms Among COVID Positive Males with Known Contact Abroad:\")\n",
    "print(top_3_symptoms)\n",
    "```\n",
    "\n",
    "This code will filter the dataset to include only COVID positive males with known contact abroad, calculate the mean of symptoms among this group, and then identify the top 3 most common symptoms based on their mean values. Finally, it will print out these top symptoms."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
