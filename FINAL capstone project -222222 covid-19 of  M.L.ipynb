{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "7aa915eb",
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
   "execution_count": 2,
   "id": "b1efa75d",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Dell\\AppData\\Local\\Temp\\ipykernel_7336\\3103726082.py:12: DtypeWarning: Columns (2,3,4,5,6,8) have mixed types. Specify dtype option on import or set low_memory=False.\n",
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
   "execution_count": 3,
   "id": "d5944ae3",
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
   "execution_count": 13,
   "id": "e468095e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(278848, 11)"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "be51dcd9",
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
   "execution_count": 16,
   "id": "ee04ddc7",
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
     "execution_count": 16,
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
   "execution_count": 17,
   "id": "42fc3d56",
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
       "      <th>Ind_ID</th>\n",
       "      <th>Test_date</th>\n",
       "      <th>Cough_symptoms</th>\n",
       "      <th>Fever</th>\n",
       "      <th>Sore_throat</th>\n",
       "      <th>Shortness_of_breath</th>\n",
       "      <th>Headache</th>\n",
       "      <th>Corona</th>\n",
       "      <th>Age_60_above</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Known_contact</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "Empty DataFrame\n",
       "Columns: [Ind_ID, Test_date, Cough_symptoms, Fever, Sore_throat, Shortness_of_breath, Headache, Corona, Age_60_above, Sex, Known_contact]\n",
       "Index: []"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[df[\"Age_60_above\"]==\"?\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "7b0f9ef7",
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"Age_60_above\"].replace(\"?\",df[\"Age_60_above\"].mode()[0],inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "ceb32834",
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"Age_60_above\"]=df[\"Age_60_above\"].astype(\"string\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "489a7cc2",
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
      " 8   Age_60_above         151528 non-null  string\n",
      " 9   Sex                  259285 non-null  object\n",
      " 10  Known_contact        278848 non-null  object\n",
      "dtypes: int64(1), object(9), string(1)\n",
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
   "execution_count": 23,
   "id": "9f2fb67d",
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
       "      <th>Ind_ID</th>\n",
       "      <th>Test_date</th>\n",
       "      <th>Cough_symptoms</th>\n",
       "      <th>Fever</th>\n",
       "      <th>Sore_throat</th>\n",
       "      <th>Shortness_of_breath</th>\n",
       "      <th>Headache</th>\n",
       "      <th>Corona</th>\n",
       "      <th>Age_60_above</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Known_contact</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>11-03-2020</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>negative</td>\n",
       "      <td>&lt;NA&gt;</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Abroad</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>11-03-2020</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>positive</td>\n",
       "      <td>&lt;NA&gt;</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Abroad</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>11-03-2020</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>positive</td>\n",
       "      <td>&lt;NA&gt;</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Abroad</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Ind_ID   Test_date Cough_symptoms  Fever Sore_throat Shortness_of_breath  \\\n",
       "0       1  11-03-2020           True  False        True               False   \n",
       "1       2  11-03-2020          False   True       False               False   \n",
       "2       3  11-03-2020          False   True       False               False   \n",
       "\n",
       "  Headache    Corona Age_60_above  Sex Known_contact  \n",
       "0    False  negative         <NA>  NaN        Abroad  \n",
       "1    False  positive         <NA>  NaN        Abroad  \n",
       "2    False  positive         <NA>  NaN        Abroad  "
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "3be84603",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The unique values in Cough_symptoms is: \n",
      "[True False nan]\n",
      "----------------------------------------------------------------------------------------------------\n",
      "The unique values in Fever is: \n",
      "[False True nan]\n",
      "----------------------------------------------------------------------------------------------------\n",
      "The unique values in Sore_throat is: \n",
      "[True False nan]\n",
      "----------------------------------------------------------------------------------------------------\n",
      "The unique values in Shortness_of_breath is: \n",
      "[False True nan]\n",
      "----------------------------------------------------------------------------------------------------\n",
      "The unique values in Headache is: \n",
      "[False True nan]\n",
      "----------------------------------------------------------------------------------------------------\n",
      "The unique values in Corona is: \n",
      "['negative' 'positive' 'other']\n",
      "----------------------------------------------------------------------------------------------------\n",
      "The unique values in Sex is: \n",
      "[nan 'male' 'female']\n",
      "----------------------------------------------------------------------------------------------------\n",
      "The unique values in Known_contact is: \n",
      "['Abroad' 'Contact with confirmed' 'Other']\n",
      "----------------------------------------------------------------------------------------------------\n"
     ]
    }
   ],
   "source": [
    "for i in ['Cough_symptoms', 'Fever', 'Sore_throat',\n",
    "       'Shortness_of_breath', 'Headache', 'Corona', 'Sex',\n",
    "       'Known_contact']:\n",
    "    print(\"The unique values in {} is: \\n{}\".format(i,df[i].unique()))\n",
    "    print(\"-\"*100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "4461e8b2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Column 'Age_60_above' does not exist in the DataFrame.\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "# Assuming df is your DataFrame containing the dataset\n",
    "# Check if 'Age_60_above' exists in the DataFrame before attempting to drop it\n",
    "if 'Age_60_above' in df.columns:\n",
    "    # Drop the 'Age_60_above' column\n",
    "    df.drop(columns=['Age_60_above'], inplace=True)\n",
    "    print(\"Column 'Age_60_above' has been successfully dropped.\")\n",
    "else:\n",
    "    print(\"Column 'Age_60_above' does not exist in the DataFrame.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "ca5bf8ac",
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
       "      <th>Cough_symptoms</th>\n",
       "      <th>Fever</th>\n",
       "      <th>Sore_throat</th>\n",
       "      <th>Shortness_of_breath</th>\n",
       "      <th>Headache</th>\n",
       "      <th>Corona</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Known_contact</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>negative</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Abroad</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>positive</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Abroad</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>positive</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Abroad</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Cough_symptoms  Fever Sore_throat Shortness_of_breath Headache    Corona  \\\n",
       "0           True  False        True               False    False  negative   \n",
       "1          False   True       False               False    False  positive   \n",
       "2          False   True       False               False    False  positive   \n",
       "\n",
       "   Sex Known_contact  \n",
       "0  NaN        Abroad  \n",
       "1  NaN        Abroad  \n",
       "2  NaN        Abroad  "
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "4ac2ad89",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2020-03-11 00:00:00\n"
     ]
    }
   ],
   "source": [
    "from datetime import datetime\n",
    "\n",
    "date_string = \"11-03-2020\"\n",
    "date_format = \"%d-%m-%Y\"\n",
    "\n",
    "try:\n",
    "    parsed_date = datetime.strptime(date_string, date_format)\n",
    "    print(parsed_date)\n",
    "except ValueError:\n",
    "    print(\"Date string doesn't match the specified format.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "22fd9ec3",
   "metadata": {},
   "outputs": [],
   "source": [
    "  df['Stamp'] = pd.Series(stamp) df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1bb9514f",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.drop('Test_date',axis=1)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "80adb2e4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Cough_symptoms         0.090372\n",
       "Fever                  0.090372\n",
       "Sore_throat            0.000359\n",
       "Shortness_of_breath    0.000359\n",
       "Headache               0.000359\n",
       "Corona                 0.000000\n",
       "Sex                    7.015650\n",
       "Known_contact          0.000000\n",
       "dtype: float64"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "(df.isnull().sum()/df.shape[0])*100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "b6f8f58f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 8 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "r=3\n",
    "c=3\n",
    "it=1\n",
    "for i in df.columns:\n",
    "    plt.subplot(r,c,it)\n",
    "    sns.countplot(x=i,data=df)\n",
    "    it+=1\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "853f2349",
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
       "      <th>Ind_ID</th>\n",
       "      <th>Test_date</th>\n",
       "      <th>Cough_symptoms</th>\n",
       "      <th>Fever</th>\n",
       "      <th>Sore_throat</th>\n",
       "      <th>Shortness_of_breath</th>\n",
       "      <th>Headache</th>\n",
       "      <th>Corona</th>\n",
       "      <th>Age_60_above</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Known_contact</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>11-03-2020</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>negative</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Abroad</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>11-03-2020</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>positive</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Abroad</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>11-03-2020</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>positive</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Abroad</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>11-03-2020</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>negative</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Abroad</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>11-03-2020</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>negative</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Contact with confirmed</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
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
       "4    False  negative          NaN  NaN  Contact with confirmed  "
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "96b07529",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Ind_ID                  0.000000\n",
       "Test_date               0.000000\n",
       "Cough_symptoms          0.090372\n",
       "Fever                   0.090372\n",
       "Sore_throat             0.000359\n",
       "Shortness_of_breath     0.000359\n",
       "Headache                0.000359\n",
       "Corona                  0.000000\n",
       "Age_60_above           45.659284\n",
       "Sex                     7.015650\n",
       "Known_contact           0.000000\n",
       "dtype: float64"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "(df.isnull().sum()/df.shape[0])*100\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "cd6dd137",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Corona\n",
       "negative    260227\n",
       "positive     14729\n",
       "other         3892\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Corona'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "47c63617",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Known_contact\n",
       "Other                     242741\n",
       "Abroad                     25468\n",
       "Contact with confirmed     10639\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Known_contact'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cdce1d8c",
   "metadata": {},
   "outputs": [],
   "source": [
    "DATA preproocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "a31929e3",
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
   "execution_count": 13,
   "id": "f6c707f3",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Dell\\AppData\\Local\\Temp\\ipykernel_4980\\3103726082.py:12: DtypeWarning: Columns (2,3,4,5,6,8) have mixed types. Specify dtype option on import or set low_memory=False.\n",
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
   "execution_count": 14,
   "id": "a622fc9d",
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
   "execution_count": 15,
   "id": "74115069",
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
   "execution_count": 99,
   "id": "f987cb41",
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
   "execution_count": 100,
   "id": "126b6111",
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
   "execution_count": 101,
   "id": "807ec61b",
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
     "execution_count": 101,
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
   "execution_count": 102,
   "id": "9aa09fe2",
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
   "execution_count": 103,
   "id": "9cf020a1",
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
   "execution_count": 104,
   "id": "183014fb",
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
   "execution_count": 105,
   "id": "965da277",
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
   "execution_count": 106,
   "id": "bc48a240",
   "metadata": {},
   "outputs": [],
   "source": [
    "# converting the datatype as categorical\n",
    "\n",
    "for i in df.columns:\n",
    "    if i=='Ind_ID' or i == 'Test_date':\n",
    "        pass\n",
    "    else:\n",
    "        df[i] = df[i].astype('category')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 107,
   "id": "c4d9a4dd",
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
   "execution_count": null,
   "id": "5912af8d",
   "metadata": {},
   "outputs": [],
   "source": [
    "EXPLORATORY DATA ANALYSIS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "4a62c741",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(278848, 11)\n",
      "Ind_ID                  int64\n",
      "Test_date              object\n",
      "Cough_symptoms         object\n",
      "Fever                  object\n",
      "Sore_throat            object\n",
      "Shortness_of_breath    object\n",
      "Headache               object\n",
      "Corona                 object\n",
      "Age_60_above           object\n",
      "Sex                    object\n",
      "Known_contact          object\n",
      "dtype: object\n",
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
      "4    False  negative          NaN  NaN  Contact with confirmed  \n",
      "        Ind_ID   Test_date Cough_symptoms  Fever Sore_throat  \\\n",
      "278843  278844  30-04-2020          False  False       False   \n",
      "278844  278845  30-04-2020          False  False       False   \n",
      "278845  278846  30-04-2020          False  False       False   \n",
      "278846  278847  30-04-2020          False  False       False   \n",
      "278847  278848  30-04-2020          False  False       False   \n",
      "\n",
      "       Shortness_of_breath Headache    Corona Age_60_above     Sex  \\\n",
      "278843               False    False  positive          NaN    male   \n",
      "278844               False    False  negative          NaN  female   \n",
      "278845               False    False  negative          NaN    male   \n",
      "278846               False    False  negative          NaN    male   \n",
      "278847               False    False  negative          NaN  female   \n",
      "\n",
      "       Known_contact  \n",
      "278843         Other  \n",
      "278844         Other  \n",
      "278845         Other  \n",
      "278846         Other  \n",
      "278847         Other  \n"
     ]
    }
   ],
   "source": [
    "# Basic info about data\n",
    "print(df.shape)\n",
    "print(df.dtypes)\n",
    "print(df.head())\n",
    "print(df.tail())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 116,
   "id": "9ecde8fd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjUAAAH8CAYAAAApGl55AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAABUN0lEQVR4nO3deXRUVbr//0+KgElAAzSC2u3QGkgUggnzpC3RSCtjA7YoouAVFNBGvCAoNCjKZOtSAaVtUNJXUOiA6GVwvC2iCAgN4gSYqGgUmYIJGQhk2L8//KZ+FASyC05yKqfer7WyFjln53n22WfXqYeqM0QYY4wAAABqOJ/bHQAAAHACRQ0AAPAEihoAAOAJFDUAAMATKGoAAIAnUNQAAABPoKgBAACeQFEDAAA8gaIGQMgLhXuEhkIfAJwaRQ2A0zJ+/HjFx8ef8iclJeWM86Snp2vmzJkO9FiaPXu24uPjg/qbQ4cOady4cdq8ebMjfQBQdSLd7gCAmmnEiBEaMGCA//fnn39eX331lebMmeNfVqdOnTPOM3fuXLVr1+6M45yu7du36/XXX1ffvn1d6wMAOxQ1AE7LRRddpIsuusj/e8OGDVWnTh0lJSW51ykAYY2vnwBUma+//lp33323WrVqpVatWmnkyJHKysoKaPPyyy/rj3/8oxITE3XVVVfpkUceUX5+viQpJSVFP/30k5YvX674+Hj9+OOP1rmPHDmi6dOnq3PnzkpOTtZDDz2kI0eOnNAuPT1dffv2VVJSklq2bKnevXtr9erVkqSNGzfq9ttvlyTdfvvtGjRokP/v3nvvPfXt21eJiYnq3LmzHn/8cRUWFgY9RgCcQ1EDoEp89913GjBggLKzszVjxgxNnTpVWVlZuuWWW5SdnS1JWrVqlWbOnKmBAwfqxRdf1MiRI/XGG2/o8ccflyTNmTNH5557rv7whz9oyZIlaty4sXX+sWPHasmSJRo6dKieeeYZ5ebmKi0tLaDNokWLNGnSJF177bV64YUX9Le//U21a9fW2LFjtXv3bjVv3lyTJk2SJE2aNEmTJ0+WJK1YsUIjR47UpZdequeee0733nuv/vd//1cjRozghGLARXz9BKBKzJkzR1FRUUpLS1O9evUkSR07dtR1112n+fPna9y4cdq4caN++9vfauDAgfL5fGrXrp1iYmL0yy+/SJKuuOIK1alTRw0bNgzqa62MjAy9/fbbmjRpkgYOHChJuuqqq9SzZ09lZmb622VlZenOO+/UyJEj/ct+97vfqW/fvtqyZYt69OihuLg4SVJcXJzi4uJkjNGTTz6pq666Sk8++aT/7y655BINHjxYH3zwga655prTHTYAZ4CiBkCV2LBhg9q3b6+oqCiVlJRIkurVq6c2bdro448/liR16NBBS5YsUd++fXX99dfrmmuuUc+ePRUREXFGucuvVLr22mv9y3w+n7p16xZQ1IwfP16SlJeXp127dmnXrl1av369JKm4uLjC2N9++6327Nmju+++279dktS2bVvVq1dP69ato6gBXEJRA6BK5OTkaPXq1f7zU47VsGFDSdKNN96osrIyvfLKK5ozZ46effZZ/fa3v9V///d/q3v37qedOzc3NyBPuXPPPTfg9x9++EGTJk3Shg0bFBkZqUsvvdR/yffJvkbKycmRJD366KN69NFHT1i/b9++0+43gDNDUQOgSpx99tnq1KmThgwZcsK6yMj//9DTo0cP9ejRQ3l5efroo480b948jR07Vm3atFGTJk1OK3eDBg0kSQcOHNAFF1zgX15ekEhSWVmZhg0bptq1a+tf//qXrrjiCkVGRiozM1P/+7//e9LY55xzjiTpwQcfrPBS89jY2NPqM4Azx4nCAKpEu3btlJmZqcsvv1yJiYlKTExUixYtlJaWpnfffVeSdP/99+vee++V9GsRdMMNN2jEiBEqLS31f+Lh8wV/mOrQoYMk6a233gpY/v777/v//csvv+i7775T//791bJlS3+htXbtWkm/Fj2SVKtWrYAYl156qX7zm9/oxx9/9G9XYmKizjvvPD311FP66quvgu4vAGfwSQ2AKlF+c767775bt9xyi8466ywtWbJE7733nmbNmiXp1+Jj8uTJmjlzpq6++modOnRIc+bM0SWXXKKEhARJv34y8tVXX+mTTz5Ry5YtFRUVVWnuiy++WDfffLOefvpplZSU6PLLL9cbb7yhnTt3+tv85je/0W9/+1stWrRI5513ns455xx99NFH+uc//ylJOnz4sKRfiy1JWrNmjWJjY5WQkKDRo0dr0qRJqlWrlrp27apDhw7p+eef1969e9W8eXNHxxGAvQjD9YcAHDB+/Hh98skn+ve//+1f9uWXX+rpp5/Wli1bZIxRs2bNNGzYsIATeF9++WUtXrxYP/74o6KiotSxY0eNHTtWv/3tbyVJK1eu1LRp05SXl6cFCxaoTZs2Vv0pLS3Vc889p6VLlyo3N1dXXXWVmjdvrmeeecZf3OzYsUNTp07VF198oTp16iguLk733HOPpk2bpmbNmunZZ59VWVmZxo4dq3fffVcXXXSRVq5cKUlavXq15s+fr4yMDMXExKhVq1a6//77g34MAwDnUNQAAABP4OsnADVGaWmp1c3tjj0RGUD44JMaADXGoEGD9Mknn1Ta7thzZwCED4oaADXGt99+q4KCgkrbJSYmVkNvAIQaihoAAOAJ3KcGAAB4AkUNAADwBIoaAADgCWF13WN2dp44gwgAgJohIkL6zW/Otm4fVkWNMaKoAQDAo/j6CQAAeAJFDQAA8ASKGgAA4AkUNQAAwBMoagAAgCdQ1AAAAE+gqAEAAJ5AUQMAADyBogYAAHgCRQ0AAPAEihoAAOAJFDUAAMATKGoAAIAnUNQAAABPiHS7A27w+SLk80UELCsrMyorMy71CAAAnKmwK2p8vgjF1o9RZK3AD6lKSsuUm1NIYQMAQA0VlkVNZC2fRi3eqsx9+ZKkuMb19OyAZPl8ERQ1AADUUGFX1JTL3JevL3cfcrsbAADAIZwoDAAAPIGiBgAAeAJFDQAA8ASKGgAA4AkUNQAAwBMoagAAgCdQ1AAAAE+gqAEAAJ5AUQMAADyBogYAAHgCRQ0AAPAEihoAAOAJFDUAAMATKGoAAIAnUNQAAABPoKgBAACeQFEDAAA8gaIGAAB4AkUNAADwhEi3OxDKfL4I+XwR/t/LyozKyoyLPQIAACdz2p/UHDx4UKmpqdq4caN/2bZt23TTTTcpOTlZKSkpSk9PD/ib5cuXKzU1VUlJSerbt6+2bt3qX1daWqqZM2eqU6dOSk5O1vDhw7Vv3z7/+uzsbI0YMUJt2rRR+/btNXXqVJWUlJxu9yvl80Uotn6MGjSo6/+JrR8TUOQAAIDQcVpFzX/+8x/dfPPN+uGHH/zLcnNzNWzYMPXp00ebNm3S1KlTNX36dH322WeSpI0bN+qxxx7TjBkztGnTJvXq1UvDhw/X4cOHJUlz587VunXrtGzZMn344YeKiorSxIkT/fHvv/9+xcTE6MMPP9TSpUu1fv16paWlncGmn5rPF6HIWj6NWrxV3Wd9qFGLtyqylo+iBgCAEBV0UbN8+XKNGTNGo0ePDlj+zjvvqH79+ho4cKAiIyPVsWNH9ezZU4sWLZIkpaenq3v37mrdurVq166twYMHq0GDBlq9erV//dChQ3X++eerXr16mjBhgtauXausrCx9//33+uSTTzR27FhFR0frwgsv1IgRI/yxq1Lmvnx9ufuQMvflV3kuAABw+oIuarp06aJ3331XN954Y8DyjIwMNWvWLGBZXFycduzYIUnKzMw86fq8vDzt2bMnYH2jRo0UGxurnTt3KiMjQ/Xr11eTJk386y+77DLt3r1bhw4dCnYTAACABwV9ovC5555b4fKCggJFR0cHLIuKilJhYWGl6wsKCiRJMTExJ6wvX3f835b/XlhYqHPOOSfYzQAAAB7j2CXd0dHRKioqClhWVFSkunXrVrq+vEApP7/m+PUxMTEnrCv/vTw+AAAIb44VNc2aNVNGRkbAsszMTDVt2lSS1LRp05Ouj42NVZMmTZSZmelft3//fuXk5KhZs2Zq2rSpcnJydODAAf/6b775Ruedd57OPvtspzYBAADUYI4VNampqTpw4IDS0tJUXFysDRs2aMWKFerXr58kqX///lqxYoU2bNig4uJipaWlKTs7W6mpqZKkvn37au7cucrKylJ+fr6mTZumdu3a6aKLLtIll1yi1q1ba9q0acrPz1dWVpaef/559e/f36nuAwCAGs6xm+81aNBAL730kqZOnapZs2apYcOGmjhxojp06CBJ6tixoyZPnqxHHnlEe/fuVVxcnObNm6f69etLkkaOHKmSkhINHDhQBQUFat++vZ555hl//FmzZmnKlCm69tpr5fP51KdPH40YMcKp7gMAgBouwhgTNrfIPXAgT7Vq+dSgQV11n/Whvtz965VTzS84R6v+cpV++aVAJSVlkqTIyMB2FbUBAABVJyJCatTI/jQTnv0EAAA8gaIGAAB4AkUNAADwBIoaAADgCRQ1AADAEyhqAACAJ1DUAAAAT6CoAQAAnkBRAwAAPMGxxySEK58vQj5fRMCysjKjsrKwuVEzAAAhgaLmDPh8EYqtH6PIWoEfeJWUlik3p5DCBgCAakRRcwZ8vghF1vJp1OKtytyXL0mKa1xPzw5Ils8XEVDU8IkOAABVi6LGAZn78v0Px6wIn+gAAFD1KGqqQTCf6AAAgNNDUVONKvtEBwAAnD4u6QYAAJ5AUQMAADyBogYA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TvXq1UubNm3SOeec42LvAACA22rM108FBQWKjo4OWFb+e2FhcHccBAAA3lNjipqYmBgdPnw4YFn573Xrnt7HVAAAwDtqTFHTtGlT5eTk6MCBA/5l33zzjc477zydffbZLvYMAACEghpT1FxyySVq3bq1pk2bpvz8fGVlZen5559X//793e4aAAAIATXmRGFJOnDggKZMmaKNGzfK5/OpT58+GjNmjGrVquV21wAAgMtqVFEDAABwMjXm6ycAAIBToagBAACeQFEDAAA8gaIGAAB4AkUNAADwBIoaAADgCTXmKd1n6siRI1q9erV27NihwsJC1a1bV02bNlVqamrQD8O0jeVkTtt4bvSN8Ti98XA6Xk3fp17I6WS7UO6bG+Pr1nGmurcTZyYs7lPz6aefavjw4WrQoIHi4uIUFRWloqIiZWZmKjs7W/Pnz1diYqK//akm3rfffmsVy8mc55xzjlW80tLSau8b43F64+Hkdga7DW7kDMV55HROJ9s5PXdr+jxy47VsW4y4dQwMpp0XinJbYVHU9O/fXz169NDgwYNPWJeWlqY333xTS5YskVR5AdSoUSPddNNNlcZyMuf8+fP16KOPVhqvtLTUKifj4f54OLmdwWyDGzmdHLdQzulkO6fnrtPtqnseufFati1G3DoGhnKBXFVFqBUTBpKSkkxJSUmF64qLi01ycrL/9379+pkFCxZU2HbBggUmISHBKpaTOf/85z9bxbPNyXi4Px42sWy3M5htcCOnTV435pHTOZ1s50bOYNpV9zxyY07++c9/tmrn1jHQtp2TsdzKGYywOFH43HPP1eeff17hui1btqhJkyb+37/55hsNGjSowra33XabjDFWsZzMmZGRYRXPNifj4f542MSy3c5gtsGNnDZ53ZhHTud0sp0bOYNpV93zyI05mZGRYdXOrWOgbTsnY7mVMxhhcaLw8OHD9V//9V/q3r274uPjFR0drcOHDyszM1MrVqzQ5MmT/W3LJ15SUtIJcbZs2aJGjRpZxXIyZ5MmTTRs2LBK45WUlFR73xiP0xsPJ7czmG1wI2eoziOnczrZzum563S76p5HbryWjy9GQmnu2vatSZMmKi0tdSyWWzmDERbn1EjSxo0btXTpUmVmZqqgoEDR0dFq2rSp+vXrp44dO/rbLV++XI8//vgpJ17jxo2tYjmZs2fPnlbxbHMy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b1jHcjKn7TacTFX2zQvjMXv2bC1ZskSlpaXq2LGjMjIyNGXKFB05ckTPP/+8ateurUGDBln1zenttO3bmeYNdmyd3qe22+nkvnI6p+142MTLzc11dB/U9HlkjKmS+bF8+XI1bNhQV111laZMmaJVq1YpNjZWt956q+68806rdgUFBSHbt/J2TsZyK6etsChqVq9ere+//16tWrUKWF5WVqaNGzcGfIdX2Qts4cKFysvLqzSWkzlt4x04cMAqJ+MRmHP58uVatGiRDh48qFtvvVVvvfWW/7H3zZs313333ec/2Dg1Hrbbads32/F1amyDaWeT95///KeWLVtW6XY6ua+czmk7HjbxfD6fo/ugps+jH3/80ar/wcwPp4ranJwcpaenh2TfateurZycHEeLLjdyBsXtj4qqQ2FhoXnwwQfN7bffbvbu3etf3rZt2xPa3nDDDeaKK64wt912W8BP8+bNzW233WZuvfVWq1hO5hw0aJBVPNucjEdgzvLni5SVlZnmzZsHPIuktLTUtGnTxvHxsN1O277ZbqtTYxtMTpu85TfZqmw7ndxXTud0cr45vQ9q+jyy7X8w86Nr165m165dZsuWLSYhIcHs2rXLv+7rr7823bp1s2oXHx8fsn3r1q2bo7HcyhmMsChqyi1btsx07drVvP/++8YYEzCJytm+EG1iOZ3TNp4bfavJ49GjRw/z8ccfG2OMKSoqMmVlZf51CxYsMDfffHPQfXNqO4Ppm01ep8fWqW213U4n91VV5LQZj2DiObUPavo8qop95VShlJCQELJ9a9OmjeMFoRs5gxFWRY0xxmRmZpqePXuaxx9//JQDZvNCtI3lZE7beG70raaOx+rVq01iYqIpKCgIWH7LLbeYNm3amK1bt55W35zYztPpm01eJ8fWiW213U4n91VV5nRyvjmxD2r6PKqKfeVUoXT99deHbN9uvvlmxwtCN3IGI+yKGmN+HbwJEyaYli1bnrKdzQvRNpaTOW3judG3mjoeO3bsOGHZm2++aX7++ecz6psT23k6fbPJ6+TYBtPuZHltt9PJfVWVOZ2cb07sg5o+j5zeV04WSqHcN6cLQjdyBiPCmNN4tncNdeTIEX333XcqLCxU3bp19fvf/1516tSp9G8ee+wxrVixQtu2bQs6lpM5beO50TfG4/TGw8ntPJ28buQ8VV4v5HSyXSj3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9phleww0xrmiKz4+3mruBnucsSnebd47bAtfJ/9TbFtEB1Po2QqLouamm24yr7zySoXr/vnPf5oBAwb4f7d5Iaanp5ubb775lDl79+4dMPmNMeabb74xrVu3NsuWLQs4GNkeDCvLe/3115uvvvrKGGPMt99+GxDnzTffNL169Qq6b5UdJOLj461ytmjRwrz++usmMzPT3HfffQEHk/Xr15tu3boFjIcxp37jstmnx98D4ocffjDPPPOMufHGG018fHzAC8Zmnx7bN2NO/iZSVlZmZs+ebW644Qb/2JTvw2PZvNHccccd5h//+IcxxphJkyaZ1157zb9u/vz5ZuDAgcYYY2644Qazbds2s3Xr1hM+9fjoo4/842vzBmKbM5jxrexNKSEhweo1GkzOygrf+Ph4qzGznbu249anTx+zdOlS/7p33nnH/6nQ9u3b/Qd8m3kUzOvKJp5t32zeoG1j2cxd22OW7THQGPsCrrLXjO3cNSb440xlxbvNe4dt4evkf4pti+hgCj1bYVHUfPrpp6Zdu3amW7du5r777jPjxo0zf/nLX8yNN95o2rVrF1CR2r4Qx48fb/bs2XPSnG+99Za58sorzYwZMwKWr1mzxrRo0cJcfvnl/mW2B8PK8i5cuNC0a9fOFBQUBCy///77TWJionn//feD7ltlB4mrrrrKKuekSZNM3759TYsWLUxCQoK54447jDHGLF682Fx55ZVmyZIl/r+1eeM62T694YYb/Pv02P+hHO+XX34xn3/+ecCyyvapMcG9iaxfv9507drVLFiwoMKixpjK32i2b99uOnToYG655Rbz17/+1bRo0cLceuutpn///qZly5b+/3WtXLnSJCUlmcTERHPbbbeZ8ePHm8mTJ5uHHnrIJCUlmYULFxpj7N5AbHOeanwPHjwYML6VvSm1bNmy0v1ZWc7j92llhe/1119vNWanmrtJSUn+uWs7bmvWrDHNmzc3d9xxh7nnnntMYmKiefPNN01GRoZJSkoyixcvDujrqeaRbd9s4x3ftxYtWpy0b5W9Qdtup83ctT1m2R4DjbE/1lf2mvnDH/5g/f5ijN1xxrZ4t3nvsD1mOf2fYtv/3NkWerbC4kRhSTp06JDefvttZWZmqqCgQNHR0WrWrJlSU1NVv359f7sPPvhAI0eOVJs2bRQdHa1169bpiSeeUFxcnG666SaNHz9eN998s1XOL7/8Urt371ZqamrA8k8//VT/+Mc/9Pzzz0uSduzYoSFDhuj3v/+94uLitHz5crVs2VJHjx7V119/rZdfflktW7b0/31+fr4KCgpUt25d1atXLyD2+++/r65duwYsS0tLU9u2bdW8efMT+taxY8eAWJ9++qleeOEFzZ07V9KvJx/PmjVL77//fsDJYKNHj9b//d//adasWTLGqGvXrgH9Wrp06Qk5Jeno0aPasWOHdu3apZSUFG3cuFHGGF133XX+NsnJySe91DEnJ0c//vijWrRoccI+jYmJUVxcnK6//nrVr19fkydP1qOPPlrpfqpIXl6eoqOjFRkZGbB88uTJ+uKLL/T111+rpKRE7du3V1pampYsWaIZM2booYce0p///Gd/++zsbD344IPavHmztm3bdtJ8GzZs0MMPP6zbb79dzz//vD755BP/ul9++UWvv/66PvvsM+Xm5qpOnTq67LLL1K9fP1166aX+dhkZGfr+++/VpUsXHTlyRLNnz9bu3buVkpKi/v37S5JWrVqliRMnqrS0VFdeeaV+97vf6ayzztLRo0f15ptvasyYMRo4cKBVzmPH91RzUpLefvttjRs3TrfccovGjRvnX/7BBx/o3nvvVWlpqTZs2HDK/Xl8zsps27ZNw4YNU4MGDdSsWTP/s2YyMzN14MABvfTSS6pTp06lY1aufO7m5+erU6dOeu+99yQpYO6Wj9u6detUXFwsSUpMTFTfvn0D9tVXX32lN998U2VlZbruuuuUnJys3Nxcff/99wGv9fKxzcvL07hx47Rt27YK55FN3451qnlZ3rfc3Fz16tVLCQkJKisr065du07o20MPPaT7779fTZo0qTCP7XZmZGRo165duvLKK1VaWqpZs2bp0KFD6tq1q38/2B5PbY+Btsd6m9dMz549rd5fbNkeA23eO5YtW2Z1zLI51l9zzTVWr+WvvvrKv/xUx7Zylc2joDhWHoWwgwcPmnvuuce0adPG3HHHHebrr78OWH98VfzFF1+YJ5980sycOdNs2bLFGGNMTk6O2bZtmzl48KC5++67/bEyMjIqjHV8zpO1O7aPL730krn//vvNkCFDzN13322eeOIJ88033xhjfv0o8cUXX/R/L1n+c80115g5c+aYsrIy88svv1j1zSbWsf7973+fMKYLFiwwX3zxhXWs49uV/2+jopyTJk06Id/xjhw5YmbPnm0WLlxoCgsLzV133WXi4+NNYmKimTJlijl69Kj1vioqKjKzZ882ixYtMocPHzZ33XWXSUhICIhVUf5t27aZdevWGWN+Penw3XffPWnO3bt3B+SsyIEDB8ydd95pWrZs6V9mO3ePbTd48OBTzredO3ead955xxQWFpqcnBzz2GOPmeHDh5v09PSgcpaUlAQ1jz7//HPz9ttvn7DdW7duNcOGDat0f5b3zWaflsvNzTX/+te/zLRp08yECRPM448/bv71r3+ZX375xXrMfvnlF6vxOHaOx8fH+3+OH4/yvOVf9VWUt6LXVXm8Y2PZ9s12Xh6/DZXt01Ox2c7jcx67rcfmtD2e2h4Dy53qWH+sU71mgnmN2vbN5hhYrrL3jnInO2Yd61TH+mOd6rV8zz33nLC8omNbVQmLT2rGjRunnJwc3XzzzXrrrbf0wQcfaNGiRYqLi5N06qr4dGM5mVOSpk2bpvXr12v48OGKi4tTdHS0/3+dc+fO1dVXX60DBw5Y5bSJNXbsWMf6NXbsWEdzStLjjz+ujRs36ujRo2rcuLEiIiI0ZswYHT16VE888YS6dOmi3bt365dffql0PGxi/eUvf7HqlxP7/eeff9b5558fVDw35vj06dP18ccfO7JPbffB+PHjrfapk9tp2852jtvEu+mmmxyLVRXbsGnTpkrHd+nSpY7ltD22OX3cteHWe0JNdOyxTZLVPAr2Pj9hUdR06dJFq1atUmxsrCTp6aef1sqVK/Xaa68pNjY24F4PlQ3yvffeq3feeafSWE7mlKS//OUvSk9P1+9+97sT1mVlZWnAgAGKiIiwytmxY8dKY61bt86qbyNGjNDy5csrjeVkTunXj0Rff/11HTx4UL1799batWt17rnnSpJ++ukn/70pbMajS5culcb6v//7P6u+2c6PYLbTdhts2jmZ08l9eu+992rVqlWV7oNQfi3bjodNvLPOOsuxWFWxDddff72ysrJOevfXiIgI/eY3v3Esp+2xzenjbtu2bUP2PcGmb7ZsY9m0i4iIsLorcNu2ba3m0fbt2yuNdazIypvUfMXFxQHf848ePVrffvutHnjgAb344osBAzphwoRTDrIxxiqWkzkjIiJUr149NW7cuML1DRs2VGlpqXXfSkpKKo0VzHjYxHIyZ0REhGJiYtSoUSM1atRIjRs39h8oJKlx48bKy8vzj1u5k43H4cOHK40VzHg4ud/POeccR+ebkzmdnkc2+yCUX8u242ETz8lYVbENixcv1oABAzR69GjdcMMNFbZv3769YzndOu5u3749ZN8TbPoWERGhCy+80LFYTue0mUdBc/bbrNA0ZMgQM3v27IDvg/Py8ky3bt38Z9iXy87ONqmpqWb16tVnFMvJnMYYM2LECDN27Fizf//+gOXZ2dnmgQceMH/5y1+sc9rEsu2bbSwncxrz65UtFd3Aqbi42MyYMcN/7wSb8bCJZds3p/e70/PNyZxO7lPbfRDKr2Xb8bCJ52SsqtgGY369d0nXrl1Petm8kzndOu7atAvlvjkdy+mcxlQ+j4IVFkXN9u3bTZcuXczQoUMDln///ffmmmuuOeEGP6caZNtYTuY05tdJctttt5mEhATTvn1707VrV9OhQweTkJBgBg4caLKzs61z2sSy7ZttLCdzGmPMxx9/bFq2bHnCpZt//OMfTdeuXU1mZqb1eNjEsu2b0/u9KuabUzmd3Ke2+yCUX8u242ETz8lYVbEN5ZYvX24OHDhwwtg6ndOt465Nu1DuW1XEcjqnMaeeR8EKi6LGmF+vcPn2229PWJ6bm2sWLFhwwvJTDbJtLCdzlvv+++/Ne++9Z9544w3z7rvvml27dp1W32xiBdM321hO5jz+AGvMrzdzOvaN0XY8bGLZ9s3p/V4V882pnMY4t09t90Gov5ZtxsM2npOxqmqfnoqTOd087lbWLpT7VhWxnM7ppLA4URgAAHifz+0OAAAAOIGiBgAAeAJFDQAA8ASKGgA1wq5du6ot1759+1RYWFht+QA4g6IGgKMmTZqk5ORkJScnKzExUQkJCf7fk5OTtXnz5qBjfvXVV+rRo8dp92njxo2Kj4+3anvgwAF169ZNBw8ePO18ANwRFncUBlB9pkyZoilTpkiSXnvtNc2ZM0f//ve/zyhmXl6e/6nXVa2oqIhPaYAaik9qAFSbH374Qffcc4/at2+vrl276umnn9bRo0clSfn5+Ro9erTat2+vzp0767/+67/0zTffKCsrS0OHDpVk/9C/ffv26Z577lGrVq107bXX+p9BVe7f//63BgwYoI4dO+rKK6/Ubbfdpl27dqm0tNT/iVCPHj20evVqSdKqVavUs2dPtW7dWn379tVHH33k5LAAcAhFDYBqUVhYqMGDB6tp06Zau3atXnnlFX388ceaPXu2JOmll15Sfn6+PvjgA73//vs699xz9eSTT+rCCy/UvHnzJElbt25VcnJypblGjx6tyMhIrV27VgsXLtTatWv96/bs2aNRo0Zp2LBhWr9+vdasWSNjjJ577jnVqlVLK1eulCStXLlSN954oz744ANNnjxZkyZN0ieffKL77rtP9913nzIyMqpglACcCYoaANVizZo1Onr0qB544AGdddZZOv/88zVq1CgtWrRIkhQVFaUdO3bo9ddf1969ezVt2jTNnTs36Dw//fSTNm/erDFjxqhevXo6//zzde+99/rXN2zYUKtWrVJKSory8/O1Z88eNWjQQHv37q0w3sKFC3XLLbeobdu2qlWrlrp27aqUlBQtXrz49AYCQJXhnBoA1eKnn37SwYMH1bZtW/8yY4yKi4uVnZ2toUOHqk6dOlq6dKmmTJmiCy+8UP/93/+t66+/Pqg85cXJBRdc4F920UUX+f9du3ZtrVy5UosXL1ZERISaNWum/Px8RUZWfDj86aef9Mknn+jVV1/1LystLVWHDh2C6heAqkdRA6BanHfeebrooov01ltv+Zfl5+crOztbDRs21M6dO5WSkqLBgwcrLy9Pr7zyikaPHq0NGzYEnUeSsrKydNlll0n69Suncm+++aYWLlyoV199VRdffLEk6bHHHtPXX3990nh9+vTRsGHD/Mt2796tqKiooPoFoOrx9ROAatG1a1cVFBRo/vz5Onr0qA4dOqRx48Zp9OjRioiIUHp6uh588EFlZ2erXr16qlevnmJiYlSnTh2dddZZkn69CqoyF1xwgbp06aLp06crNzdX+/fv15w5c/zr8/Ly5PP5FBUVJWOM1q5dq9dff91/dVV5rvz8fEnSn//8Z/3P//yPPvvsM0nS559/rr59+/rPvQEQOihqAFSLevXqKS0tTRs3btTVV1+t6667Tj6fz3/ezAMPPKCLL75Y3bt3V6tWrfTaa6/p+eef11lnnaVmzZqpdevWuuqqq/TBBx9Umuupp57S2Wefra5du6pfv37q1KmTf92f/vQnderUSd27d1eHDh00d+5c3XHHHfruu+909OhRNWrUSKmpqbr55pv16quv6o9//KMeeOABPfzww2rVqpVGjRqlwYMHa9CgQVU2VgBOD0/pBgAAnsAnNQAAwBM4URhAjTJy5Eh9/PHHJ13/6KOPqlevXtXYIwChgq+fAACAJ/D1EwAA8ASKGgAA4AkUNQAAwBMoagAAgCdQ1AAAAE+gqAEAAJ5AUQMAADyBogYAAHjC/wculzwLao1CzwAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "# Bar plot for categorical variables\n",
    "for col in df.columns:\n",
    "    if df[col].dtype == 'object':\n",
    "        df[col].value_counts().plot(kind='bar')\n",
    "        plt.title(col)\n",
    "        plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 117,
   "id": "d68aa86f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABv4AAAJOCAYAAAB/dnBOAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAAChpElEQVR4nOzdfVxUZf7/8Td3xo0soBhaWZY4WN6siAuiZqVRraaQ4k2RqYWWsLX6XaxMW10N1N22jEory6jVsgVjC1fL2nWTTJCMtDVBwFLKvAEFuRG5m98f/uasI6CgIDi+no/HPB7O+ZzrXNfMnDN8PJ8517Ezm81mAQAAAAAAAAAAALis2bf2AAAAAAAAAAAAAABcPAp/AAAAAAAAAAAAgA2g8AcAAAAAAAAAAADYAAp/AAAAAAAAAAAAgA2g8AcAAAAAAAAAAADYAAp/AAAAAAAAAAAAgA2g8AcAAAAAAAAAAADYAAp/AAAAAAAAAAAAgA2g8AcAAAAAAAAAAADYAMfWHgCuPD/99JOGDx9eb8zBwUFXXXWVOnXqpFtuuUWjR4/WsGHD6l03PT1dDz30kCRp9+7dcnS8uN3ZbDZr37596t69e5PaTZo0Sdu3b9djjz2mWbNmSbJ+jZs2bdINN9xwUWNrrJMnT6qwsFDXXXedsezDDz/UnDlz5OPjoy1btlyScVxKX3zxhZYvX669e/dKkq699lqlpKTIzs6uUe1ramr0+eefa+PGjdq1a5eOHj0qOzs7XX311fL391d4eLiCgoJa8iW0qJdfflmvvPKK+vfvr/fff7+1h2Oz8vLydNNNNzV6vwMANC/yy5ZDftm4/NLymTXWq6++qjvvvPOixwoAuLyRw7QccpimnSMrLCzU+++/ry1btujAgQMqLS2Vh4eHrrvuOg0ePFgTJkyQj49PS7+EVlddXa38/HzdeOONxrLmPr5a24Ue37i8XN57KS57JpNJ7du3N57X1taqpKRE+fn52r9/vzZu3KghQ4Zo2bJlcnd3b7Fx7Nq1S4sWLdINN9yg559/vsX6aUkpKSn6y1/+oscff1zjxo1r7eFcErt27dJjjz2m2tpaeXp66rrrrpOPj0+jiy/79u3TrFmzlJWVJUlydXVVt27dVF1drZ9++kkff/yxPv74Y91zzz1avHixXF1dW/Ll4DJUWlqqF154QR988IF27tx52Sd/AGALyC+bD/ll0/PLjh07NuqEpqen50WOFABga8hhmg85TNNymH//+9968sknVVJSIicnJ3Xr1k033HCDTpw4oe+++07ffvutVq1apT/+8Y8aM2bMJXg1rePLL7/Uc889p7vvvtsoXNsaWzi+0TicoUSrmjdvXr1XU1VWVuqjjz7S4sWL9eWXXyoqKkpvvfWW2rVrZ6zTt29fbdiwQZIu+mT7e++9p127dl3Qr46WLl2qkydPysvL66LGcLFefPFFHT58uM7ykJAQ/frXv5aTk1MrjKplffrpp6qtrdV1112nf/7zn3J2dm5024yMDE2fPl3l5eXq3bu3nnjiCd12221GvKKiQmvXrtVLL72kTz75RMePH9eqVaso7MDK7t27tWbNmtYeBgDgDOSXzYf8smn5pSQNHTpUS5YsaaHRAQBsGTlM8yGHaXwOk5eXp5kzZ+rUqVP63e9+p4cfflhubm5G/NixY/rrX/+qpKQkzZ07V507d9agQYNa6mW0qtdff10//PBDaw+jRV3M8Y3LC/f4Q5vUrl07jRs3Tq+//rocHBy0fft2vfPOO1bruLi4qHv37q1+WfI111yj7t27q0OHDq06joa4u7ure/fuuv7661t7KM3u+PHjkqRf//rXTTopc/z4cf3hD39QeXm5hgwZovfee8+q6CdJzs7OmjJlil599VXZ2dkpPT1dq1evbtbxAwCAS4f8svmQXwIAcOmQwzQfcpi6Vq1apVOnTmnEiBF6/PHHrYp+ktShQwctWrRIwcHBqq2t1SuvvNKs4wbQMij8oU37zW9+owkTJkiS3nzzTVVUVLTyiNCW1NbWSpLVr9waY9myZTp8+LBcXV31l7/8RVdddVWD6w4aNEi//e1vJUlvv/220ScAALg8kV/iXC40vwQAoKWRw+BcLjSH+e677ySdLhg2xN7eXuHh4cb6nBsD2j4Kf2jzLElNUVGRvv76a2N5enq6/Pz85Ofnp+rqaqs2O3fu1KxZs3T33Xerb9+++s1vfqPx48frjTfeUGlpaZ1tJCcnSzo9B7ifn58mTZok6fQNiP38/DR48GAdPnxY06dPV9++fRUYGKg//OEPkk7fuNjPz08vvvhiveOvra3V3/72N917773q06ePBg0apN///vfGH9Yznes1WVji6enpkqSXX35Zfn5++vnnnyWdnhrCz89PL7/8sqTTNy728/PT0KFD693etm3b9Pjjj2vIkCHq3bu3Bg4cqMjISG3atKne9YcNGyY/Pz/l5eVp+/btmj59uoKCgtSnTx/99re/VXx8vMrKyuptey41NTVKTEzUpEmT9Jvf/Ea9e/fW7bffrtmzZ2v37t1W61pes+VzS05ONt6Xn3766Zz9nDp1Sh9//LEkKTw8vFG/QouOjlZ8fLzWrVsne3vrr82mjFv63z7l5+en/fv319uf5T3+8MMP68T27dunZ555RsOHD1efPn00bNgwLV26VCUlJca+aNk3znbs2DEtWbJEISEh6t27twYPHqwnnnii3n3xYuTl5WnOnDkaOXKk+vXrp4CAAIWFhenFF19UYWGhsd6JEyfUt29f+fn5Nbi/SdKUKVPk5+enN954Q9L/Pv8XX3xRR48e1fz58zV06FD16dNHd955p1588UVVVlZKOn1MPfLII/rNb36jvn376r777tM//vGPOn08/fTT8vPzU0JCgvLy8vS73/1OQUFB8vf3V1hYmN555x2dOnXKqs2wYcOMmztLUq9everdBy/0GNu/f7++/PJLTZkyRQMGDFD//v01ceJE/ec//5F0+obTb775pu6991717dtXQUFB+t3vfqe8vLw626ypqdF7772nSZMmGeMYMmSIoqOjtXnz5gbfewCwVeSX1sgvLy6/bC6ff/65pk+fruDgYPXu3Vu33nqr/vCHP9QZa35+vnr27Ck/Pz99//33DW7v7rvvlp+fnxITE+u0X7BggUJCQtSnTx8NGDBADzzwgBITE1VTU1NnO5Y86f3339cHH3yg22+/XX369NFdd93VYN4JAGgZ5DDWyGEuPoexTHv6n//8R2azucH1hg0bpo8++kifffZZnXNjknT48GEtXbpUI0aM0K9//Wv5+/srNDRUr7zyik6cOFFnfcu4n3/+eX3++ee6++671bt3bw0bNkz//Oc/jfUqKyv1zjvvaMKECQoICFDfvn119913a/HixTpy5Mh5X19jWPaL7du3S5Jee+01+fn56emnn66zbnV1td555x3dd999+vWvf63f/OY3evDBB42pds9k2YfHjx+vvLw83X///erTp4+Cg4O1dOlSq3U//fRTRUZGauDAgcY5m8cff1zbtm1rcNxHjhzRiy++qPDwcAUGBqpXr14KDAzUxIkTtWrVKqsfB5zv+IbtofCHNq9nz57GzY0tX8DnsmnTJj3wwAPasGGDiouL5evrqw4dOmjXrl3661//qokTJxqJjbu7u/r376+OHTtKOn35ev/+/WUymay2WVlZqUceeURfffWVunfvLjs7O1177bWNGv+zzz6r5557TkePHpXJZFJlZaU++eQTjR8/XuvWrWvKW1GvLl26qH///sYvem644Qb1799fXbp0OW/bRYsWacqUKdq0aZOqqqrUs2dPOTk5KTU1VY8//rhmzpypqqqqetsmJibqoYceUlpamnx8fNShQwft27dPr776qh555JF6Txo0pLS0VBEREZo3b562b98ud3d3+fn5qaSkRB9//LHCw8P19ttv13nNls+tY8eO6t+/v/r373/Oq/ckKTMzU+Xl5ZKkwYMHN2p8vr6+uvvuu+Xt7X1R475YqampGjt2rNatW6djx44Z+9OqVas0btw4Y1qH+hw5ckT33XefMZ4bb7xRxcXF+vTTT3X//fefM5FoiszMTIWHh+vDDz/UkSNHdOONN8rHx0d79+7Va6+9pvvuu0+//PKLJOlXv/qVQkJCJEkfffRRvds7dOiQ0tPT5eDgoNDQUKvY/v37FRoaqr///e/y8PCQt7e38vPz9dprr+npp5/W+++/r8mTJ+ubb77Rddddp6uuukrff/+9nnrqKa1du7be/rKzszVu3Dh9/vnnuvrqq9W5c2ft2bNHcXFxmjp1qkpKSox1e/fubfVdUd8+eDHH2DvvvKNHHnlEu3fvVteuXY3397HHHtNnn32mhx9+WH/5y1904sQJ3XjjjTpx4oQ+++wzPfDAA1b3MjCbzZo1a5b+9Kc/afv27XJ1dZWfn59qamr0+eef67HHHtNLL73U4GcKALaI/PLcyC+bll9erOrqasXExCg6OlpffPGF7Ozs5Ofnp8rKSq1fv17jxo2zmnK+a9euCgwMlCTjB21n+/bbb/Xjjz/KxcXFmL1Ckj777DPde++9ev/993XkyBHddNNN6tChg3bs2KF58+bpkUceafAE5ccff6w//vGPMpvN6tatm44ePaqbb765Gd8JAMD5kMOcGzlM03OYIUOGSDpd9Jw0aZI+/fRT47zZmVxdXdWzZ0917ty5Tmzbtm0aOXKkVq1apQMHDujGG2/Utddeq7179+rll1/W6NGjlZ2dXW//GRkZeuKJJ3TixAl1795dR44cMfKL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6+rUqZMOHjyov/3tb/WO4amnnjLO3Vjyi3/961/KzMzU1VdfrTfffFM33XST1ZhjY2P161//WsePHzdyqUthxowZCgsLM/YRNzc3/fGPf5R0+n379ttv62336KOPysfHx2hz1VVX6dChQ8aMVIsWLdI999xjrO/g4KCIiAjjfN8rr7xizMqQlZWloqIitWvXTs8995w8PDyMdk5OTpo+fboxc9TevXub8dXjckLhD5cFy6X0dnZ251zPy8vL+LKLiYlRZmam1Q1nx48fr1dffVXjx49v8hgCAgKa3EaSHnzwwTrL7OzsjDF88cUXF7Tdi/H111/rxIkTcnR0NAoJZxsxYoR8fHxUU1Nj3FPsTLfffnu9n4flD/GZUyKeiyWh6du3r/r371/vOg8//LCk01M7XuwfLBcXF+PfDU3R0BiXetyff/65pNPz+dc3l/oDDzxgJM716dSpU73jtCShZrP5vP9paIwbbrhBkjR//nxt27bN6j0eNmyYXnvtNT366KPGMjs7O+OE1NnTfe7cuVP79u2Tp6enhg0bVqcvT0/POjefdnFxMZLE+ubsv/rqq41/n3kvA4uBAweqZ8+edZYPGTJE1113nWpraxt1T7zmOMaGDh1a57M+s2h622231WljeX1nvjZLYfSLL77Q66+/bky1avHSSy/pz3/+c50TkQBg68gvm9eVnF+e6czptRp6nOmLL75QZWWlfH191atXr3q3aZnufNeuXcb9ks+cwjMlJaXONouKinTttdcaJwwrKyu1ZcsWSdLo0aPr7cfPz089e/aU2WyuN9+55ZZb5Orq2ti3AgDQQshhmhc5jOTh4aHFixdr8+bNmjdvnm677TarYm5FRYU+/vhjjRgxQl999ZVVW8v7cf/991v9APzMbY8dO1bS/85tnalTp05GkepMlnXvvPPOevMPOzs7I6dpzHma5nLXXXfVWebj42MU0I8dO1Zvu/qOmS1btqi6ulqdOnXSiBEj6m334IMPysnJSSUlJcb0vr169VJGRoYyMjLk5eVVp01lZaVx7J88ebJRrwu2x7G1BwA0huUP5K9+9atzrufg4KCYmBg9++yz+uKLL/TFF1/Iw8NDQUFBGjx4sG6//fZ656JujE6dOl1Qu4b+A285yX706FGdOHHivK+tOe3bt0/S6SJNQ7/KsrOz0y233KLDhw/rhx9+qBM/s4ByJsvVSY2dv9wylobeJ+l04aJ9+/YqLS3VDz/8cFHzT5857uPHjxu/lGuqSznuyspK/fjjj5JUb1FKOv1roRtvvLHBpK+hz+vM5OnMm/5eqNmzZ2vGjBnauXOnpkyZIldXV/3mN7/RoEGDdPvtt9c7P/qYMWP06quvGveisaxjudpv1KhR9SaPDc3Rf/avxM7k6Pi/P3vmem5afeYVkWez3Bjb8lmcS3McY/V9V51Z3D3f67MYNmyYAgMDtX37dr3wwgt64YUXdNNNN2nQoEG69dZbFRwc3OL3LgKAtoj8snldyfnlmYYOHaolS5Y0ev2cnBxJp+9rfP/999e7zpk5y759+4x7s4wdO1aJiYnatGmT5s+fb7xPlh9T3XfffcZJyB9//FGVlZWSpD/96U/15laSjF/yW97DM13o/goAaF7kMM2LHOZ/fHx8NGnSJE2aNEk1NTX6/vvv9dVXX2n9+vXau3evTp48qejoaP3rX/9Shw4dVFpaasyK1Lt37wa3a3k9TXnvLOe3Nm/ebMxkdLYTJ05IOp3nmM3m8xbDm4Plqr2zubm5qaioqMHZtOo7Ziyf980331zvj/yl0+ftLOf7fvjhB91xxx1GzNnZWT/++KP++9//6sCBA8rPz1dubq6ys7ONcZxZ7MeVhcIf2rzKykr9/PPPkqTu3bufd/3x48frhhtu0Ntvv62vvvpKxcXF2rRpkzZt2iQ7OzvdfvvtWrBgQZOTG8sf66ZwcnJq8D/VZxacTp48eUmTmjNv3HwuloTHcin5mRp6XRb1FVUuZixubm4qLS2tdyxNceONNxr/zsnJ0XXXXXfeNrW1tcrOzpafn5/xh/hSjvv48ePGv8/1K+tzTa1xqQo7Q4cOVVJSklauXKn//Oc/KisrM/6DsXjxYgUEBGjhwoXGlCCSdM0112jgwIH66quv9PHHH+uJJ55QZWVlvVNUnenMqzfr01DSdC5nTo9wNst7b0ksz6U5jrHz/aK+sa/P0dFRb731ltasWaMPP/xQe/fu1b59+7Rv3z6tXr1a7du3V2RkpB577LFLkiQDQFtAftn8ruT88mJYTt6Wlpbqm2++Oe/6Z+Yh/v7+uummm7Rv3z79+9//1ogRI1RcXKz//Oc/srOzU1hYWJ1+JOm///1vo8d1Jn4oBACtjxym+ZHD1M/BwUF9+vRRnz59NH36dL377ruKi4tTeXm5PvroI02dOtWq/3Odk7LEysvL6xToGsovLO/FL7/8UmfmorPV1NSorKys0VPOXowLzYfqO2YuZt/buXOnnn/+eeMqQAsvLy/ddttt+v777+udlhZXDgp/aPN27dplTGPQ0GXuZwsKClJQUJAqKir09ddfKyMjQ6mpqdq9e7c2b96sX375Rf/4xz9a/CR3VVWVKisr600AzvzPdH0JTX1JQXl5ebOMy5JQnW+qAcuJhQu9Kq45x2KJX+xYbr75Zl177bX6+eeftXXrVqtfyjRk586dmjhxojw8PPT222+rV69ezTLuhhK/sz/nMwtA9U1PadGaJ63OdPPNN+uFF15QVVWVdu7cqfT0dH311Vf65ptvtGPHDk2ZMkWbNm2yel1jx47VV199pZSUFD3xxBPGFFU9e/bULbfccsnGfq5jzPLeW35lfy5t6RiTTv8nZOrUqZo6daoOHTqktLQ0paena8uWLSooKNCyZcvk7OysqVOntug4AKCtIL/8H/LLlv87fC6WHzLdfffdio+Pb3L7++67T3/961+VkpKiESNGaOPGjaqsrFRgYKDVtFln5l3ffPNNq75mAMCFI4f5H3KYix/L3//+dyUkJBj34muInZ2dJk+erI0bNyozM9O4cu/M/s91vqq4uFjS6XyksfuZJUd69tln650i1hZc6L6Xl5enhx56SBUVFfL19dXYsWPVs2dPde/e3bgiceLEiRT+rnDc4w9tXmJioqTTl37/5je/Oee6lZWVysvL086dOyWd/jXFkCFDNGvWLH344Yd64YUXJJ2+CWp2dnbLDvz/q2+aHEn6/vvvJUnXX3+98cfszBstW6biOdORI0eaZUyWOcb379/f4B/m2tpaY4yW+7a1BMtYdu/e3eA6eXl5RkLXHGOxzJudnJxs3CflXNasWSPp9BVWPXr0kHTh4z5zKsb6PuOKioo6f/Dd3d2NaS0b2m9PnTrVqCkoW1JNTY3279+vjIwMSad/zTdgwABFR0drzZo1WrNmjezs7HT06NE6c8KHhITIw8NDBw4cUFZWlj777DNJDV/t11Is023VxzK1xJlXKzakLR1jxcXF+vbbb41fyHXu3FlhYWFavHix/vOf/xjF77PvsQgAtoz88n/IL1t2LOdjmY3iXDnIyZMntX37duXn59eZKiwsLEwODg768ssvVVpaatzvz3IvHYuuXbsa+0Jubm6Dfe3atUvZ2dlt5gdlAABr5DD/Qw5z8WNxcHBQXl6etm3bZjXbVEMs01Va7ivXvn17Y6rOc80oYInVd/uXhjQmR/rll1/07bffNtu+cKlZPu89e/Y0OCVnaWmpcb7P8nm/8847qqio0E033aSkpCQ9/PDDGjRokNU0pJYpWHHlovCHNm379u36+OOPJUnTp0+3+qNfny1btmjEiBGaPn16vUnBoEGDjH+f+Z9my69NGnvpfVOsW7euzrKamhq9//77kk7ff8vizBuy1pcMWYoh9WnKawgICJCHh4eqq6uNotbZ/vnPf+ro0aOys7PTrbfeet5tXihL0WHXrl0NTm+UkJAg6XTBwjLv+8V49NFH1alTJ5WWlmru3LkNzr8tnb6Z8Pr16412ll+mXei4PT09jc+qvs/43//+t6qrq+ssDwkJkXR6f6rvM05OTj7n67gUcnJydNddd2ny5Mk6evRonbi/v7/x66SzE5qrrrpK9957ryTpk08+0RdffCEnJyeNGjWq5Qd+hi1bttQ7dsuvINu1a2d1zJ453eaZn0tbOsaeeeYZTZgwQStXrqwTc3JyUmBgoKTG33MAAC535JfWyC+bJ7+8ULfddpscHBy0b98+bd26td51EhISNGnSJIWGhurkyZNWsauvvlq33nqrKisrtW7dOu3YsUNubm666667rNZr37698Tf/3Xffrbef/Px8PfDAAxo9erQ++eSTZnh1AIDmRA5jjRzm4nOY4cOHy83NTadOndKf//znc65bWFiobdu2SZJuv/32OmN+//33693PiouL9Y9//EPS6dvDNJZluxs2bGjwR/uW8x3/93//1+jtnsulvv3J0KFD5ejoqKNHjxq3uznb6tWrVV1dLRcXFyOXO3O63/pug7N161bjvs1nn+tpyeMbbQuFP7RJZWVlWrNmjR599FHV1tYqODi4wZvdn2no0KHy8vJSUVGRnnrqKRUVFVltc+nSpZKkLl26GFduSf+7VNrypdic/va3v2nNmjVGoaO0tFRPPvmkdu/eLQ8PDz388MPGujfeeKMxjeCf//xn41Jus9mslJQUvfrqqw32Y5m+x/Llfy4uLi6aPn26JCk+Pt5qfJL06aef6o9//KOk0/PBn3lfvObm7++v2267TZL0xBNPKD093YhVVlYqPj5ef//73yVJTz75ZLP8EXZ3d9eiRYvk5OSkzZs3KyIiQqmpqVZ/9EpLS7V8+XLNnDlTZrNZgwYN0kMPPXTR43Z2djamrnz55ZetfoHz5ZdfauHChfWO+eGHH5abm5syMzO1YMECq5M+n332mbFvt6aePXvKZDKppqZG//d//6dDhw4ZscrKSr344osqLS2Vq6urBgwYUKe95Zfp77zzjoqKinTHHXeoQ4cOl2z80umpQqKioqzmj09PT9ecOXMknf7P1Zlzr585bdaZ3x9t6RgLDQ2VJH3wwQf6xz/+YbWf5+Tk6G9/+5skGfszANgq8kvyS6nl8ssLde2112rcuHGSpP/7v//Tv//9byNWW1urxMREvfLKK5KkiIiIeu9dY5kh4aWXXpLZbNY999xT772CH3/8cTk4OGj9+vVavHix1VV9e/fu1fTp01VVVaVrr732kv/4CgDQMHIYchipZXIYT09Po2j24YcfKjIyUrt27bI6b1BbW6tt27bpoYceUklJiUaMGCF/f38jPm3aNLm5uWnv3r36/e9/b1Wky8/P16OPPqqCggL5+Pho8uTJjR7biBEjZDKZdOLECT3yyCNWV/6VlpZqwYIF+uqrr2RnZ2d8fherKftNc+jSpYvGjx8v6fSUpmf+8Kq2tlbvvfeeXn75ZUlSVFSUcT7Ksg9u3bpVX3/9tdGmurpa69ev16xZs4xlFRUVVn225PGNtoV7/KFVPffcc1b/ea2urtaJEyesprEZNmyY/vKXv1hNkdiQdu3a6aWXXtIjjzyiDRs26F//+peuv/562dvbKz8/X+Xl5XJxcdGSJUus5hS/+eabJZ2+38U999wjX19f4z/YF8PJyUlDhgzRwoULtWLFCvn4+Gjfvn0qLy+Xm5ubXnrpJavLsO3t7TVz5kw9++yz2r59u2677TbdeOONOnLkiI4ePao77rhDx44dM6ZpONMtt9yivXv36s0339SWLVsUEhKiqKioBsf2yCOP6KefftL777+vhQsX6uWXX1bXrl116NAh4xL5u+++W3Pnzr3o9+F8/vznP+uxxx5TZmamHnroIV177bXq0KGDfvjhB5WWlsrBwUEzZ87UyJEjm63PO+64Q2+99ZaeeOIJfffdd4qMjNSvfvUrXXfddaqurtYPP/xgzJt/7733KjY2ts6v6S503DNnztSMGTOUm5urO++8U76+viouLtbPP/+sPn36qH///vrXv/5l1aZLly7685//rJkzZ2rt2rX6+OOP1b17dxUWFurgwYO6+eablZubq6qqqvP+6q8lvfjii5o4caK2b9+uO++8U9ddd51cXFz0008/6cSJE3JwcNDChQvrLej16tVLPXv2NKbUvNTTfEqnp53Ys2eP7rzzTplMJpWXlxtTKtx777169NFH66zv6uqq8vJyjR8/Xtddd51iY2PVs2fPNnOM3XXXXRo/frz+/ve/66mnntLSpUvVpUsXlZaW6sCBAzKbzerbt68ee+yxFh0HAFwq5Jfkl1Lr5JcX6plnntHhw4e1efNmzZgxQ1dffbV8fHz0888/69ixY5JOv28zZ86st/0dd9whT09P44RuQzlUQECAFi1apPnz5yshIUFr165V9+7dVVZWpv3798tsNsvb21tvvfVWvfdfAgC0LHIYchjp0ucwDz74oKqrq7Vs2TKlpqYqNTVVXl5exi1nfv75Z+Mefffcc4+WLFli1b5r166Kj4/X73//e/373//WbbfdJl9fX9XU1Cg3N1e1tbW65ppr9MorrzTpx91OTk5avny5IiMjtWfPHt1777268cYb5eLioh9//NGY7nTOnDlNupLwXG655RZt3rxZKSkpys7O1oABAzR//vxm2XZD5syZo8OHD+tf//qXfv/73+vqq69W586dlZ+fb0y/+uCDD2ratGlGm4cffljr16/X8ePHFRERoW7dusnNzU0//fSTiouL5erqKn9/f2VmZlr9KF9queMbbQ+FP7SqvXv3Wj23t7eXi4uLbrrpJvXp00ejR49WcHBwk7YZFBSkxMREvf3229qxY4d+/PFHOTo6qnPnzhoyZIgefvhhXXPNNVZtwsLCtG/fPn300Uf66aefVFVV1eDcyk1hZ2enl19+WStXrlRycrKys7Pl5eWlESNG6LHHHlPXrl3rtBk/frw6d+6shIQE7dq1S3l5ebrxxhv16KOPKiIiQhEREfX29dRTT+nkyZP66quvtG/fPuXl5Z13bAsWLNCdd96p999/X99++6327NkjLy8v3XHHHQoPD9edd9550e9BY3h6eupvf/ubkpOT9fHHHys7O1tHjx6Vj4+P7rnnHkVERBhXyTWnoKAgbdq0SUlJSfriiy+Ul5ennJwcOTg46JprrlFAQIDGjh1b79VpFzPuoUOH6r333tMbb7yhr7/+Wrm5ubruuuv0xBNPaNq0aXrmmWfq7e/OO+/UunXr9NprrykjI0NZWVny8fHRo48+qkcffVRBQUGSVO9l/peKr6+vkpOT9dZbb2nbtm06ePCgzGazrr76aoWEhGjq1KlWvyQ823333afFixerU6dOLTp9RkP69Omj559/XvHx8dqxY4ccHR0VGBio+++/37g35Jks/zn561//qn379umnn37STz/9pJ49e7apY+xPf/qT/P399Y9//EPZ2dnKzs6Wm5ubAgICNGLECI0fP15OTk6XZCwA0NLIL8kvpdbLLy/EVVddpRUrVmjjxo368MMP9d///ld79uyRm5ubgoKCNGbMGI0ePdpqivEztWvXTqNGjdLf/vY33XDDDQ3mrtLpGRb69eund955R1999ZVycnJkZ2en7t276/bbb9fDDz9sXF0BALi0yGHIYaTWyWGmTJmie+65R8nJyfrqq6904MAB7du3T/b29urUqZOGDx9+zv1vyJAh+uc//6m3335bX3zxhX744Qc5OTnp5ptv1j333KOJEyfqV7/6VZPH1bVrVyUnJ+v999/Xp59+qry8PFVUVMjLy0u33nqrJk2adN57XTbFtGnTdOTIEX322Wf68ccfrWZ8aint2rXTq6++qo0bN2rdunVGHtipUycNGjRIEyZMMM73WVxzzTX6+OOPtXz5cm3btk2//PKLHBwc1KVLF40ePVqTJ0/Wzz//rMmTJys9PV3l5eXG1YwNHd8N5Zm4fNmZmdAVAC575eXlxlQLW7ZssfqV3OVkyZIlevvtt/XII4/oySefvGT9Pv3000pOTtaoUaP0/PPPX7J+AQAAAAAAAKA5UcoFgMvA3LlzFRoaqg8//LDe+BdffCFJ8vb2vmyLfqdOndLHH38sOzs74143AAAAAAAAAIDGo/AHAJcBPz8/ZWVl6YUXXtD3339vFfv666+1aNEiSdIDDzzQGsO7YMePH1deXp7y8vI0a9YsFRYW6vbbb2/Rm2UDAAAAAAAAgK3iHn8AcBmYMGGC1q9fr507d+q+++7TddddJ09PTxUUFBg36r377rv16KOPXlQ/SUlJWrdu3QW1jY+PV6dOnZrU5scff9TEiRON5y4uLnr66acvqH8AAAAAAABceb7//nvjR/FN9dhjj+m2225r5hEBrYvCHwBcBq666iqtXr1a69evV3Jysvbv32/c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ZN68eRkwYEC6du2a5s2bZ9iwYSktLc3ChQvrxocPH56DDjoo++yzT66++uo8+uijqaioyKpVq7J06dKMHTs2xcXFOeSQQzJy5Mi65wYAAAAAAIDGrFlDF/B2NTU1ee6551JcXJzbb7891dXV6d27d8aMGZMVK1bk6KOPrjf/qKOOyvz585MkK1euzODBg98xvnz58mzZsiUvv/xyveXLysqy33775fnnn0+SlJSUpF27dnXjRx55ZNauXZvXXnst++67725vQ5Mme7zZwF7kM8n7adffk7+rjx7vOQAAAACNXaML/l599dUce+yx6d+/f6ZPn56NGzdm3LhxGTt2bA444IAUFxfXm9+qVats27YtSVJZWflnxysrK5MkrVu3fsf4rrG3L7vr/rZt2/Yo+Nt//7a7PRfYu0pL2zR0CRQo3/UAAAAAQGPT6IK/srKyeqfXLC4uztixY3PuueemvLw827dvrzd/+/btadOmTd3cdxsvLS2tC/F2Xe/v7cvX1ta+Y2zX/V3Pv7s2bNiS2to9WmS3NW1aJMiAPbBxY2Wqq2saugwKSJMmb4Z+e/O7nsZp13sPAAAAAI1Vowv+li9fngcffDCjR49Ok/85p9aOHTtSVFSUTp065c4776w3f+XKlenQoUOSpEOHDlmxYsU7xk877bTst99+adeuXVauXFl3us9169Zl06ZNOfroo1NTU5NNmzZl/fr1KSsrS5K88MILad++fdq23bOdfLW1sTMYGhGfR/YG3/UAAAAAQGNT1NAFvF1JSUnmzp2b22+/PTt37szatWszbdq0nH322enfv3/Wr1+fOXPmpKqqKosXL86CBQvqrus3ZMiQLFiwIIsXL05VVVXmzJmTDRs2pF+/fkmS8vLyzJw5MxUVFdm6dWumTJmSHj165NBDD83hhx+erl27ZsqUKdm6dWsqKioyY8aMDBkypCFfDgAAAAAAANgtjS74a9++fW677bb89Kc/TY8ePTJ48OB07NgxEyZMSGlpaWbPnp0f//jH6dmzZ8aPH5/x48enV69eSZITTzwxEydOzKRJk9KjR4889NBDmTVrVkpKSpIko0aNSu/evTN06ND07t07b7zxRr71rW/VrXv69OnZuXNn+vbtm3PPPTennnpqRo4c2QCvAgDAe7dw4cIce+yx6dKlS91t7NixSZKnnnoq55xzTrp06ZI+ffpk3rx59Za9//77069fv3Tu3Dnl5eVZtmxZ3Vh1dXVuvPHGnHTSSenSpUtGjBiRV155pW58w4YNGTlyZLp165aePXvm+uuvz86dOz+YjQYAAACg8Z3qM0l69OiRu++++13HOnbs+GfHkmTgwIEZOHDgu441b948Y8aMyZgxY951vKysLNOnT9/zggEAGpFnnnkmAwcOzA033FDv8c2bN+eiiy7KZZddls9+9rN54oknMmrUqBxzzDHp1KlTlixZkuuuuy6zZs1Kp06dMnfu3IwYMSI///nPU1xcnJkzZ+axxx7Lvffem7Zt2+aaa67J+PHj873vfS9JcsUVV6Rdu3ZZtGhR1q9fnxEjRmTOnDm58MILG+JlAAAAAPjIaXS/+AMA4G/zzDPP5BOf+MQ7Hn/44YdTUlKSoUOHplmzZjnxxBNz1llnZe7cuUmSefPmZcCAAenatWuaN2+eYcOGpbS0NAsXLqwbHz58eA466KDss88+ufrqq/Poo4+moqIiq1atytKlSzN27NgUFxfnkEMOyciRI+ueGwAAAIC9T/AHAFBAampq8txzz+WRRx7J6aefntNOOy3XXHNNNm/enBUrVuToo4+uN/+oo47K8uXLkyQrV678s+NbtmzJyy+/XG+8rKws++23X55//vmsWLEiJSUladeuXd34kUcembVr1+a1117bi1sMAAAAwC6N8lSfAAC8N6+++mqOPfbY9O/fP9OnT8/GjRszbty4jB07NgcccECKi4vrzW/VqlW2bduWJKmsrPyz45WVlUmS1q1bv2N819jbl911f9u2bdl33313exuaNNntqcAHwGeS99Ouvyd/Vx893nMAgA+G4A8AoICUlZXVO71mcXFxxo4dm3PPPTfl5eXZvn17vfnbt29PmzZt6ua+23hpaWldiPf666+/6/K1tbXvGNt1f9fz767992+7R/OBvae0dM8+v7C7fNcDAMDeIfgDACggy5cvz4MPPpjRo0enyf8cWr9jx44UFRWlU6dOufPOO+vNX7lyZTp06JAk6dChQ1asWPGO8dNOOy377bdf2rVrV+90oOvWrcumTZty9NFHp6amJps2bcr69etTVlaWJHnhhRfSvn37tG27Zzt3N2zYktra97T5f1XTpkWCDNgDGzdWprq6pqHLoIA0afJm6Lc3v+tpnHa99wAA7F2u8QcAUEBKSkoy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sWPHZsKECXnyySdz8803Z9asWXn66adTU1OTkSNHpqioKD/5yU+yYMGCvPLKK5kwYUIDbRWwuyorK/PYY4/lyiuv/Ktzv/nNb+aRRx7JnDlzsmjRohx//PG54IIL8sYbb3wAlQI0Tnop+GjTSwH8bfRS8NGml2JPucYfvMXkyZMzZcqUeo89+uij+drXvpY2bdqkefPmWbt2bdq0aZPt27dn8+bNadOmTb35rVq1yvz581NTU5MTTjghTz75ZIqKivL000/nueeeyx133FG3zLhx43LGGWfkmmuuSWlp6Qe2ncCeGTRoUFq0aJEWLVr8xXm1tbW5++67M3369BxyyCFJklGjRuWee+7JI488kv79+38Q5QI0GL0U8G70UgC7Ry8FvBu9FHtK8AdvMXHixHc9l/pvf/vb3HTTTfnDH/6Qww8/PIcddliSpKampt68Vq1a5b/+678yY8aMjB49Olu3bk3//v0zfvz4rF69OtXV1endu3e9ZVq0aJGKigoNFjRiBx544G7Ne/XVV7Nt27ZcfvnlKSr63x/VV1VVZc2aNXurPIBGQy8FvBu9FMDu0UsB70YvxZ4S/MFfUVVVlYsvvjhf/vKXc95556VJkyZ59tln88ADD7xj7tatW/PKK6/k5ptvTpL87ne/y5e//OV897vfTb9+/dKqVassWbIkTZs2TfLmOZcrKirqGjagcWrSpEndv4uKilJVVVV3v6amJps2bUqSlJaWpmXLlpk9e3Y6d+5cN+f3v/992rVr90GVC9Co6KUAvRTAe6eXAvRS7CnX+IO/oqqqKtu3b0+rVq3SpEmTrF27NtOmTasbe6vKysoMHz48CxYsSG1tbQ488MAUFRWltLQ0nTp1ymGHHZapU6emsrIy27dvz5QpUzJs2LBUV1c3xKYB78GRRx6Z559/PitWrMjOnTtz++23Z9u2bUnebL6GDBmSm2++OS+//HJqampy//3359Of/rQLKQMfWXop4K30UgB7Ri8FvJVeit0h+IO/onXr1pkyZUq+853vpEuXLvnCF76Qk08+OWVlZfl//+//1Zvbrl27TJ8+PbNmzcoJJ5yQT3/60+nVq1eGDRuWZs2a5bbbbsv69evzT//0TznllFPy0ksv5Y477kjLli0baOuAPfXJT34yZ511VoYNG5ZTTz01GzduTNeuXevGx40bl+OPPz7nnXdeunXrljlz5mT69Ok59thjG7BqgIajlwLeSi8FsGf0UsBb6aXYHU1qa2trG7oIAAAAAAAA4G/jF38AAAAAAABQAAR/AAAAAAAAUAAEfwAAAAAAAFAABH8AAAAAAABQAAR/AAAAAAAAUAAEfwAAAAAAAFAABH8AAAAAAABQAAR/AB8SW7ZsyauvvtrQZQAAAADQSFRXV6eioqLB1m9/FTQ+gj/gA/Piiy9m3LhxOe2009KlS5d88pOfzNe//vVUVlbu9XWff/75ueWWW/b6evamfv36ZcWKFQ1dBgBAkqRPnz7p2LFjunTpUu92wQUXNHRpAAB7zebNmzNp0qT07t07nTt3zimnnJJx48bl5ZdfbpB6rrzyyvzgBz9IkqxevTrHHHNMVq9e/YGt3/4qaHwEf8AH4te//nXOPvvsHHzwwfnBD36QZcuWZdasWXnqqadywQUXpLq6uqFLbPQ2btzY0CUAANQzefLkLFu2rN5t9uzZDV0WAMBec+WVV2bjxo2ZP39+fvOb3+QHP/hBduzYkX/913/Nzp07P/B6Gnp/UUOvH3gnwR/wgZgwYUIGDRqUyy67LB/72MeSJH/3d3+Xb37zm9l///1TUVGRNWvW5IorrsiJJ56Yk08+OaNHj84rr7ySJFmyZEmOOeaYes/5la98JV/5ylfq7t911105/fTT07Nnz1x55ZX50pe+VO9XfqtWrcoFF1yQ7t27p2/fvvnxj3+8W7Xv3LkzkyZNysknn5yePXvmvPPOy5NPPpkkOfPMM/Pd73633vyzzjor8+fPz3333ZfzzjsvN954Y3r06JFevXrlP/7jP3LPPffk9NNPT9euXTNhwoS65fr06ZNbb701/fv3T5cuXTJ06NCsXLkySdK/f/8kyfDhwzNr1qwkyf/9v/835eXlOeGEE9K/f//MmTMnNTU1da/Ntddem4svvjhdunRJv3798vjjj+e6665L9+7dc/LJJ2fevHl1677lllvSu3fv9OjRI4MHD85Pf/rT3XptAADezY4dO/Ltb387ffv2TY8ePTJ8+PCsWrUqSfLtb387n/vc5+rNnzZtWi666KIkyfr16zNmzJicfPLJOeWUUzJhwoRs3bo1yZs9Ye/evTN69Oh069Yt3/ve9z7YDQMAPvKefPLJ9OvXLwcccECSpKysLFdddVWOP/74vPbaa9m4cWOuueaanHLKKenZs2cuvvji/OEPf0jyv7/Imzp1arp3757JkycnSR566KGcddZZ6dq1a8rLy/PLX/5yt2q5+uqr89///d+57bbbcskll9Q9vmDBgpx55pnp3Llzhg0blj/96U9Jkvvuuy/l5eW54IIL0q1btyxYsCDbt2/PTTfdlN69e6d79+45//zz8/TTT9c9169//et84QtfyCmnnJKOHTumvLw8v/nNb5K8+/4qoOEJ/oC97qWXXsqKFSvy6U9/+h1jZWVlmTFjRg4++OBccMEFadq0aR5++OH86Ec/SpJccsklu3W01EMPPZRbb701N998c375y1+mW7duefjhh+vNeeyxxzJ69OgsWbIk5eXl+epXv5qqqqq/+tw//OEPs2zZsvzoRz/Kr371q3qNWXl5eX74wx/WzX322WezevXqnHnmmUnebAbbtWuXxYsX57LLLssNN9yQJUuWZOHChZkzZ07mz5+fJ554om7573//+/nWt76Vxx9/PEceeWQuueSSVFVV5Sc/+UmSZNasWRk+fHgWL16cK664IhdeeGGWLl2ab3zjG7njjjty11131T3Xvffem+HDh+fJJ59Mp06d8sUvfjGHH354Hn/88Vx88cW59tprs2PHjixevDjf//73M2/evCxZsiTnnHNOrr766t16bQAA3s03v/nNPPLII5kzZ04WLVqU448/PhdccEHeeOONDBkyJE899VTdDrDq6uo88MADGTJkSGpqajJy5MgUFRXlJz/5SRYsWJBXXnml3sFSL7/8co444og8/vjjOe+88xpoCwGAj6oBAwZk4sSJmTRpUhYuXJg1a9bkgAMOyNSpU/Oxj30sl112WV566aXcf//9+cUvfpEjjjgiw4YNqzuQKUkqKyvz2GOP5corr8wvfvGLTJw4MRMmTMjSpUvzpS99KV/60pd26/SZ119/fbp165aLL7643oHpzz33XO6555784he/yObNm/Od73yn3thZZ52VX/3qV+nXr18mTZqUX/7yl7nrrrvy2GOP5ZOf/GSGDRuWtWvXZvv27RkxYkT69++fRx99NEuWLMmhhx6am266KUnesb8KaBwEf8Bet+sCv2VlZX92zn//93+noqIikydPTtu2bbPvvvtm8uTJWb58eZ599tm/uo758+fns5/9bE444YQ0b948Q4cOTceOHevN+dSnPpXjjjsuRUVF+dSnPpVt27Zlw4YNf/W5W7VqldWrV2f+/Pl58cUXc/nll+eBBx5IkgwaNCgvvfRSnnnmmSTJD37wg5xxxhlp06ZNkqR169b5l3/5lxQVFeWUU05JdXV1vvjFL6a4uDgdO3bMgQcemDVr1tSt64tf/GL+4R/+Ia1atcpXv/rV/PGPf8yvf/3rd9R03333pW/fvvnUpz6VZs2a5bjjjstFF12Uu+++u25Or1690q1btxQVFaVXr15p3bp1zj///DRr1iynn356duzYkfXr16dly5bZvHlz7rnnnvz2t7/NOeeck8cffzzNmzf/q68NAPDRNnny5HTr1q3ebdu2bbn77rvz5S9/OYccckhatmyZUaNGpaqqKo888kgOPvjgnHTSSXXXovnlL3+Z6urqnH766Xn22Wfz3HPPZeLEidlnn31SWlqacePG5aGHHqp3GqkhQ4akefPm2WeffRpoywGAj6qvfe1rmTBhQv74xz9mwoQJ6dOnT/r165cHHnggFRUVWbp0aa655poccMABadWqVcaMGZOdO3fmF7/4Rd1zDBo0KC1atMi+++6b//zP/8znP//5dO/ePU2bNs3pp5+ePn361NvHs6cuueSStG3bNvvtt19OPfXUvPTSS3VjzZs3z8CBA9OiRYs0adIkDz74YEaPHp3DDjssLVq0yL/8y7/kiCOOyIMPPpjmzZvn+9//fs4777zs2LEja9asSUlJSd0vCIHGqVlDFwAUvl2nPli3bl0OP/zwd4yvX78+GzZsSGlpab2dN/vss09KSkqyZs2avxgaJskf//jHutML7HLIIYfUu19SUlL3712h1u78mnDAgAGpqqrKvHnz8o1vfCP7779/Lrnkknz+85/PAQcckFNPPTU//OEP8/d///d58MEH651etKSkJE2aNEmSFBW9eazFvvvuWzdeVFRUd3rOJDnssMPq/l1cXJySkpKsW7fuHTVt2LAh//AP/1DvsY9//OP1QsS3bm/Tpk3rrXdXTTU1NenSpUtuueWW/Md//Eduv/32tGrVKueff35GjBhRVzMAwLuZOHFiysvL6z22YcOGbNu2LZdffnm9XqKqqqquVznnnHNy00035fLLL8/999+fgQMHpnnz5lm9enWqq6vTu3fves/ZokWLVFRU1N0/8MAD9+JWAQD8eUVFRRk4cGAGDhyY2travPDCC/nhD3+Yf/u3f8vll1+epP4+qaZNm+aggw7KmjVrcvzxxyep38usWbMmS5cuzX/913/VPVZdXZ1evXq95xrfvg+surq67v4BBxxQ16Nt3rw5VVVV+fjHP15v+Y9//ONZvXp1mjZtmiVLlmT48OHZtm1bjjrqqDRr1iy1tbXvuTZg7xP8AXvdwQcfnKOPPjoLFy5M9+7d641t2LAhp59+ei699NJs3LgxW7durQv/tmzZko0bN9ZrSHbs2JEWLVokefPiwaWlpXXrWLt2bb3nXrt2bY444oi/uf4XX3wxxx13XAYNGpTt27fnxz/+ccaNG5du3bqlQ4cOGTx4cCZPnpyTTz45bdu2rbeNuwK23fXWI6YqKyuzcePGHHTQQe+Yd/DBB9c7WitJKioq6kLWPVn32rVrs//+++ff//3fs2PHjjz++OO59NJLc9xxx+Uf//Ef96h+AIDS0tK0bNkys2fPTufOnese//3vf5927dolSfr27ZvJkyfn0Ucfzc9+9rPcf//9SZL27dunVatWWbJkSZo2bZrkzf6voqIihx12WN11lve0xwIAeD8sWrQol112WX7+85/XHex91FFHZfTo0XnsscfqDu5+6aWX0qFDhyRvhnhr1679s/ts2rdvn0GDBtVd7zh5c19Nq1at9so2vHXdZWVladmyZSoqKnLkkUfWPf7SSy+lT58+eeqpp3Ldddfl7rvvzic+8YkkyezZs/Piiy/uldqA94efcgAfiGuuuSb33ntvbr311mzcuDG1tbX53e9+l0suuSTHHXdcLrjgghx11FGZOHFitmzZki1btmTSpEk59NBDc8IJJ+TQQw9Ns2bN8tBDDyVJfvWrX2Xx4sV1z3/uuefmnnvuydNPP52dO3fm3nvvrbvQ8N/q5z//eS699NKsXr06rVq1SklJSZo1a5a2bdsmSf7xH/8x1dXVmT59+juOeN9Td9xxR1atWpXXX389N9xwQ4444oh06dIlyZtHum/ZsiVJMnjw4PzsZz/Lj370o1RXV+e3v/1tZs2alcGDB+/xOp955plceOGFWb58eVq0aJH9998/SepCVQCAPVFUVJQhQ4bk5ptvzssvv5yamprcf//9+fSnP51Vq1YlefPI80GDBmXy5Mk57rjj6nY0derUKYcddlimTp2aysrKbN++PVOmTMmwYcPqHakOANAQunfvnv333z9f/epX8/zzz6eqqipbt27NAw88kD/84Q8544wz0rt373zta1/LunXrsn379nz961+vO635uzn33HNz11135emnn07y5n6a8vLyPPjgg7tV01v3F+2poqKiDB48ON/4xjeyatWq7NixI3feeWdWrlyZAQMGZMuWLSkqKqoLIX/zm9/krrvuyo4dO96X9QN7h1/8AR+IHj165D//8z/z3e9+NwMGDMjrr7+esrKynHHGGbn44ovTvHnz3HbbbZk6dWr69++fHTt25KSTTsodd9yRZs2a5cADD8xVV12VGTNm5LrrrkuvXr1SXl6e119/PUnSv3//vPTSSxk5cmR27NiR0047LZ/4xCfel+vUfeELX8if/vSnfO5zn8vWrVtz8MEH55vf/Gbat2+f5M0dV5/5zGdy1113ZebMmX/Turp27ZpRo0Zl7dq16d69e773ve/V/drxs5/9bEaPHp1hw4blyiuvzLe//e185zvfyVVXXZXS0tJ8/vOff08XUu7fv3/+8Ic/ZMSIEdm4cWP233//XHXVVXWnnwAA2FPjxo3LLbfckvPOOy+bNm3KIYcckunTp+fYY4+tm3POOedk9uzZGTVqVN1jzZo1y2233ZYbb7wx//RP/5Q33ngjnTp1yh133JGWLVs2xKYAANRp1apV/s//+T+59dZbM2LEiGzYsCHNmzdP586dc8cdd+TII4/MTTfdlK9//es5++yzs23btnTu3Dl33nlnSkpKsnXr1nc85xlnnJFt27blqquuytq1a1NSUpJhw4bl/PPP362aBg0alEmTJuXZZ5/NTTfdtMfb9G//9m+55ZZbMmzYsGzatCnHHHNM/v3f/z1/93d/l8MPPzznnXdehg4dmpqamnz84x/P+eefn5tvvjnr169PWVnZO/ZXAQ2vSa0T8gIFYPny5Wnbtm0OPvjgusfKy8vzuc99Lueee+5eX/9dd92VRx99NLfffvt7fo4+ffrk0ksv/Zt/NQgAAAAAwEeTU30CBWHx4sW55JJLsm7dutTW1mbhwoVZuXJlTjzxxL263nXr1uXpp5/OnXfemc9//vN7dV0AAAAAAPCXONUnUBD++Z//OWvWrMnZZ5+dysrKHHHEEZk5c2YOOeSQv7jchg0b8slPfvIvzlm2bNmfHXvkkUfyta99LQMHDkzfvn3fU+0AAAAA8Hbl5eV58cUX/+z4rFmz0q1btw+wIuDDwKk+AQAAAAAAoAA41ScAAAAAAAAUAMEfAAAAAAAAFADBHwAAAAAAABQAwR8AAAAAAAAUAMEfAAAAAAAAFADBHwAAAAAAABQAwR8AAAAAAAAUAMEfAAAAAAAAFADBHwAAAAAAABSA/x+aQxrnYlSH9QAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 1800x600 with 3 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "cols = ['Cough_symptoms', 'Fever', 'Sore_throat']\n",
    "\n",
    "# Set up the figure and axes\n",
    "\n",
    "fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))\n",
    "\n",
    "# Plot univariate distributions for each column\n",
    "for i, col in enumerate(cols):\n",
    "    sns.countplot(data=df , x=col, ax=axes[i])\n",
    "    axes[i].set_title(f'Distribution of {col}' , fontsize=18 )\n",
    "    axes[i].set_ylabel('Count')\n",
    "    axes[i].tick_params(axis='x')\n",
    "    axes[i].grid(True)\n",
    "\n",
    "plt.tight_layout()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 118,
   "id": "7f11e87e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABv4AAAJOCAYAAAB/dnBOAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAACYWUlEQVR4nOzdeVyVZf7/8TegxqLBURy00lzwaLkk6oCoaWnUqLmMoliMmYUWMJV+BysTR0fDZWzKodLKdJjKlgFj0rKyZpwkciEjbUPBUklzAUFZRFnO7w9/5x6PgHIQxLl5PR+P83hw7uu+7vs693Y+XJ/7XLeLzWazCQAAAAAAAAAAAMD/NNeGbgAAAAAAAAAAAACAy0fiDwAAAAAAAAAAADABEn8AAAAAAAAAAACACZD4AwAAAAAAAAAAAEyAxB8AAAAAAAAAAABgAiT+AAAAAAAAAAAAABMg8QcAAAAAAAAAAACYAIk/AAAAAAAAAAAAwARI/AEAAAAAAAAAAAAmQOIPTvv555/VtWvXKl8333yzAgICdOedd2rGjBn697//Xe1ytm/fbtQrKyu77HbZbDbt27fP6XqTJ09W165d9dxzzxnTzv+MBw4cuOy21dTp06f1888/O0x799131bVrVw0ePPiKteNK+uyzzxQWFqaAgAAFBATo7rvvls1mq1Hd3NxcvfDCC5o4caL69++vHj16aODAgQoLC1N8fLyOHj1aZb2rdZvW9hjGpZWWluq5557T0KFD1aNHDwUHB+v555+v9fKef/55de3aVffcc08dtrLh5Ofn6/jx4w7TzPYZATQcYsf6Q+xYs9jRvs+efPLJSy7//H25ffv2ump2nWuoY+589vV/8cUXDbJ+AEDNEIvVH2Ix5/rxJKm8vFwff/yxZsyYoaFDh6pnz57q1auX7rjjDs2aNeuqjr+A/yUk/nBZrFar+vTpY7x69uyptm3b6pdfftGHH36oyMhIPfjggyooKKjXduzevVsTJ07UypUr63U99WnDhg266667tHXr1oZuyhWze/duPfzww/r666/VrFkzderUSe3bt5eLi8sl6/773//WXXfdpeeff17ff/+9fH191b17d1177bX65ptv9OKLL+quu+7Su+++ewU+yeUzwzF8NVuyZIleeuklHTp0SNdff738/Px0/fXXN3SzrgoJCQm68847lZmZ2dBNAdAIEDvWHWJH52JHAABALFaXiMWcj8V+/PFHjRs3To8++qg+/PBD5eXlqUOHDrr++ut19OhRrV+/Xvfdd58ee+wxFRcX1/OnAcytSUM3AP/bYmNjFRQUVGn62bNn9d5772nx4sX6/PPPFRUVpdWrV6tZs2bGPL169dLGjRslSU2aXN6h+Oabb2r37t268cYbna67dOlSnT59WhaL5bLacLmee+65Kn+hFhISoltuuUVNmzZtgFbVr48//lgVFRW64YYb9MEHH8jd3b1G9fbt26cZM2bozJkz+v3vf68HHnhAXl5eRvmJEyf0l7/8RUlJSZozZ47atGmjAQMG1NfHqBOXcwzj0j788ENJ0vTp0/WHP/yhgVtzdVm8eHFDNwFAI0LsWHeIHWseOwIAgHOIxeoOsZhzsVhaWpqmT5+u4uJi9ejRQ48++qiGDBlilJeUlOjtt9/WX//6V3300UfKy8vTmjVrLvtYAxorfvGHetGsWTNNmDBBL7/8stzc3LRjxw79/e9/d5jHw8NDnTt3VufOnRuoledcd9116ty5s1q2bNmg7ahOixYt1LlzZ7Vv376hm1Ln8vLyJEm33HKLU8HCmjVrdObMGY0YMUKPPPKIQ9JPklq2bKmFCxcqODhYFRUVeuGFF+q03fjfYz/WAgMDG7glAICqEDvWHWJHAADgLGKxukMsVnW9P/zhDyouLtagQYP05ptvOiT9JMnd3V3333+/XnzxRbm4uGj79u1644036rT9QGNC4g/16te//rXCwsIkSa+++qpKSkoauEW4mlRUVEiSwx1kNfHNN99IOhdoVMfV1VWhoaHG/PZ1oXGq7bEGALiyiB1xMXyfAwBQv4jFcDG1jcWWL1+uo0ePytPTU8uWLdM111xT7bwDBgzQ8OHDJUl/+9vf6M8DaonEH+qdPWDIz8/Xl19+aUy/2EOBd+3apZkzZ+quu+5Sr1699Otf/1oTJ07UK6+8osLCwkrLSE5OlnRufO2uXbtq8uTJkv77cN+BAwfq6NGjmj59unr16qXAwEBjuL+qHgp8voqKCr3++uu6++671bNnTw0YMECPPfaYkXw6X00edGwvtz+s9vnnn1fXrl116NAhSeeGXejatauef/55SZd+KPDWrVv1yCOPaNCgQerRo4f69++viIgIbdq0qcr5hw4dqq5du2rfvn3asWOHpk+frqCgIPXs2VPDhw9XfHy8ioqKqqx7MeXl5UpMTNTkyZP161//Wj169NBtt92mWbNm6bvvvnOY1/6Z7fstOTnZ2C4XPhS5KvbhEv7zn/9c9AHCQ4cO1XvvvadPPvlErq5VX+6Ki4v1wgsvaMSIEerVq5eCgoL04IMPKjU1tdrlHj16VEuXLtWIESN0yy23KCAgQGPGjNELL7ygU6dOVZrf/nmfeeYZffrpp7rrrrvUo0cPDR06VB988EGNj2GbzabExERNmDDBeIhyWFiY1q1bV+12KCws1IsvvqixY8cqICBAvXv31qhRoxQfH19lW+2f7+mnn9bo0aPVp08fBQQEaMSIEXr66aer3T8ffPCBHnzwQd1+++3q0aOHgoOD9eCDD2r9+vV1HqQ5s/3tx7vdfffd57B968KxY8f01FNPaeDAgcZ59Nxzz1W5fZ988kl17dpVb731lt555x3ddttt6tmzp+68806HB1jXZr9J0o4dOzRr1izdcccd6t27t3r06KFbb71VjzzySKXnDtjbYjd16lR17dq1yudinjhxQkuWLFFISIh69OihgQMH6tFHH63yOggAl4vY0RGx4+XHjnXp7Nmz+vvf/66wsDD17dtXvXr10l133aXFixfr2LFj1db74YcfNHfuXA0fPlx9+vRRjx49NGDAAE2bNk0fffRRtfW+//57/eEPf9CQIUPUq1cvjRo1SmvXrr1oDCw5FxOc7+DBg1q8eLF+85vf6JZbblGfPn00adIk/eMf/7hoTLd582bdf//96tevnxG3vPzyyzp79myV85eXlys5OVn33XefAgMDjdh47ty52r9//0U/GwCgfhGLOSIWu7xY7MyZM1q/fr0kKTQ0tEa/1IyOjlZ8fLzWrVtXqT/PmXZLNTum7K7kvjl27Jiee+45hYaGKjAwUN27d1dgYKAmTZqkNWvWkHTHZWOQXNS7bt26qXnz5iosLNSOHTs0aNCgi86/adMmzZw5U2VlZbJYLPL391dRUZF2796tXbt2af369Xr77bfVvHlztWjRQn369NGBAweUm5urli1bqkOHDrJarQ7LPHv2rB588EHt379fXbp00eHDh3X99dfXqP1z585VWlqafHx8ZLVadeDAAX300UfatGmTnn76aY0fP77W20aS2rZtqz59+ujbb7/V2bNndeONN6pVq1Zq27btJesuXLjQ+Nm7j4+PunXrpqNHjyolJUUpKSkaPny4li1bVuW44omJiUpISFCzZs3UoUMHnTx5Uj/++KNefPFFffHFF1q7dq3c3Nxq9BkKCwsVERGh9PR0SdL111+v9u3ba//+/Vq/fr3ef/99Pf7445o6darDZ7bvt1atWhnjul/srh+7QYMG6dtvv9XWrVs1efJkTZ48Wbfeeqs8PT0d5vP09FS3bt2qXU5JSYnCwsK0d+9etW3bVh07dtT+/fv1+eefKzU1VYsXL9Zvf/tbhzr2IKCgoEBNmzaVv7+/ysrKtHfvXmVkZCgpKUkvv/yyQ0LFLi0tTWvWrJG3t7c6d+6sffv26YYbbqjRMWyz2fTEE0/ovffe07XXXquOHTsqOztbX3/9tb7++mv99NNPiomJcaizb98+TZs2TYcOHZKbm5vatWsnd3d3ZWVl6cUXX9Q///lPrVq1ymGYjoMHD2rSpEnKzc2Vp6enbrjhBknS/v379frrrys5OVmvv/66br75ZqPO4sWLlZCQIOncvu/atauOHTumzz//3Hj9+c9/vsgerTlnt3+PHj3k5+enr776StK5B5k3b9680vatrZycHIWGhuro0aPq0KGDWrVqpczMTL300kvasGGDXn/99SqvNevXr9dXX32lNm3aqEOHDvr555910003SardfpOkv/zlL3rllVcknRvutlOnTiosLNShQ4e0adMmbdq0SQsWLDD+ievQoYP69OlTadu0atXKYbnHjh3Tb3/7Wx05ckTt27dXx44d9dNPP+njjz/Wv//9b61atUrBwcF1sj0BQCJ2vBRiR+djx7py7NgxTZ8+XT/88INcXFx03XXXycfHR1lZWUpISNA///lPrVixQn379nWo9+abb2rhwoWqqKiQt7e3brzxRpWUlOjnn3/Wli1btGXLFj388MOaOXOmQ73169frqaeeUmlpqby9vdWlSxcdOnRICxYsuOjw5c7GBHaffPKJHn/8cRUXF+uaa66Rv7+/Tp06pfT0dKWnp2vHjh1atmyZXFxcHOqtWLFCaWlp8vT0VIcOHXTs2DHt3btXzz77rLZu3ao1a9Y4dNoVFRXp97//vb744gtJkp+fn2644Qbt379f//jHP7R+/XotW7ZMd955p/M7CQBw2YjFLo5YzLlYLD09XcXFxZKkgQMH1qh9/v7+8vf3v+x2n+9Sx9SV3Ddff/21pk2bplOnTumaa65R+/bt1aRJE/38889G3PWvf/1Lr732Wo33KVCJDXBSdna2zWq12qxWq23btm01qjNq1Cib1Wq1/eEPfzCmbdu2zVhOaWmpzWaz2crLy20DBw60Wa1W26pVq2xlZWXG/N9++62tf//+NqvVanv55Zcdlv/EE09UWv6FbQ0MDLRlZmbabDab7cyZM7aCggKbzWaz/e53v7NZrVbbs88+W2U9q9Vqe+6552xnz5612Ww2W0lJiW3+/Pk2q9Vq6969uy0rK+uin+lC1W2722+/3Wa1Wm3/+Mc/HKavW7fOZrVabbfeeqvD9NWrV9usVqvt5ptvtr3xxhu28vJyo2zjxo223r1726xWq23hwoVVrsdqtdqefPJJ26lTp2w2m81WUVFhe+ONN4yyTz75pMr2V+Whhx6yWa1W28CBAx0+15kzZ2zLly83lvnxxx871LPvtyeeeKLG67LZbLZTp07ZRowY4bCPunfvbps0aZLtmWeesX322We24uLiauvbt6n9uPjPf/5jlOXm5trCwsJsVqvVNmDAAFtFRYVR9vPPPxvb9eGHH7YdP37cKDt48KBR77bbbjO2q81ms8XHxxvri46Otp05c8ZY14Xb4mLH8E033WT7+9//bpwXJSUltpiYGOM4OH95RUVFtpCQEJvVarVFRkbajhw5YpQdO3bMNn36dJvVarXdeeedttOnTxtlM2bMsFmtVtsjjzxiKywsNKYfP37c+HwPPPCAMT0rK8tmtVptPXv2rHRMJycn27p162azWq229PT0avdHTdV2+9ts1Z93tXX+Pu3fv78tLS3NKPvxxx9tw4cPt1mtVtu9997rUM++n61Wq23BggXGvrTvu9ruN/u1p1u3brakpCSH68Evv/xiXOcGDBjgUHb+tklNTa32M95+++22Xbt2GWUHDx603XnnnTar1WoLDQ2t7WYE0EgQOxI7XuhKx472fVaTetUdrxUVFUascc8999j27dtnlJ06dco2e/Zsm9VqtQUFBdmOHTtmlP3000+27t2726xWq23FihXGcWGz2Wx5eXm2xx57zDg28vPzjbKDBw/aevbsabNarbbFixcb8WNZWZnt5Zdfdjje9u/fb9SrbUxw4MAB4xh44oknHGKpzz77zNarVy+b1Wq1vfPOO8b089uwZMkSW1FRkbGtXnrpJaPs/FjbZrPZ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+usVHBysQYMG6dZbb63xXXkXUxfbvz5c7Li3P/x73759lcqqux7Vdr9J5855V1dXffnll8rKylJ2drYOHjyoPXv26MCBA8Z85/8TVhPVHbfnd3ry0GkAdY3YsW4RO54zePBgLVmy5KLz/Pzzz1X+6s/+Hb1582ZjhIQL2W/C2r9/v2w2m0MH0DXXXKPdu3dr7969xnf03r17Hb7PbTabpHPfq7/88oskqUuXLlWuq1u3bnJxcTHqnM/ZmODMmTNGnFXd87GbN29ebVuqi1HtscL5cUJmZqYk6dtvv602eWmPv4uKinT06FFdd911Vc4HAKg/xGJ1qzHHYue3Oy8vz6HPzRl10e6qjqmG2jfu7u7av3+/vv32Wx08eFDZ2dnKysrSnj17jFjI2f4b4Hwk/lDvzp49q0OHDkmSOnfufMn5J06cqBtvvFF/+9vf9MUXX+jkyZPGQ+hdXFx02223af78+U4HDuff3VpTTZs2rfKXRZJjcuj06dNXNGAoLCyUJIcheKpi/8IqKiqqVFbd57KrqhPhctri5eWlwsLCKttSV9zc3NSzZ0/17NlT06dP12uvvaZFixapuLhY7733XqWH+17qAcTnO7/dF0tk2cuKi4ur7PC5HFU9QPh85+8z+37Zv39/pV/hXej8O49uuukmrV+/Xi+//LI++eQT5efna/v27dq+fbueffZZWa1WzZs3z2Foy2XLlql///5KTEzUrl27dOjQISUlJSkpKUnXXHONJk6cqMcff/ySx9zF1MX2rw8XC1jtZVUlxao7Fmq732w2m/7+979r9erVOnbsmDHdxcVFHTt21JgxY/Tee+9ddHnVudzjFgCcRexY94gdL5+93b/88ouRlKtOeXm5ioqKjO353nvvacWKFZW+22+44QaFhobqH//4h8P084fEqurXhdK5/eHh4aHi4mKH6bWJCc7/5X5167sYZ2IFe/ySm5tbo+HHT506ReIPAK4wYrG615hjMfvQ39K5G4BuuOGGS9apqKjQnj171LVrV2Mkgrpod1XHVEPsm127dumZZ57Rjh07HKZbLBYNGTJE33//fZVDigLOIPGHerd7927jp9zV/RT7QkFBQQoKClJJSYm+/PJLpaWlKSUlRd999502b96sX375Rf/85z/rvVO/tLRUZ8+erfICfn6ne1XBQlVfuBf+Y15b9mDlUj/jt991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33yzfvnlF+Xl5enIkSMOvwK4+eablZaWpgULFuitt97Svffeq9DQ0DraQgDgiNiR2FG68rFjXWjatKlWrFihiIgI/fDDD7r77rvVsWNHeXh4aP/+/cYv7WbPnu0w8sDMmTMVFRWlrKwsDRs2zOjsOXDggM6cOaN27drJxcVFBw8edBiGs3Xr1lq+fLl+//vf67333tMnn3yizp0768iRIzp+/LiGDh2qzz77zOFO8MuJCfz9/bV06VI9/vjjWrt2rZKTk9WpUyfl5ubql19+kSSNGzdO995772Vvy2XLlumhhx7Srl27dM899+iGG26Qt7e3srOzjQ7XKVOmaNKkSZe9LgCAI2IxYjGpYWKx22+/XatXr9ajjz6qb775RhEREbr22mt1ww03qKysTD/99JNxM/ndd9+tuLi4SjcA1Ve7r+S+eeCBB/T+++8rLy9P4eHh6tChg7y8vPTzzz/r5MmT8vT0VEBAgNLT03lsCy4LiT9clr179zq8d3V1lYeHhzp16qSePXtq9OjRxlB/NRUUFKTExET97W9/086dO7V//341adJEbdq00aBBg/TAAw8Yww3ajR07Vj/++KPee+89/fzzzyotLa2TYWFcXFz0/PPPa9WqVUpOTtaePXtksVg0YsQIPfzww2rXrl2lOhMnTlSbNm2UkJCg3bt3a9++ferYsaMeeughhYeHKzw8vMp1PfHEEzp9+rS++OIL/fjjj9UOfXN+2+bPn6877rhDb731lr7++mv98MMPslgsuv322xUaGqo77rjjsrdBTfj4+Oj1119XcnKy1q9frz179uj48ePy8/PTb37zG4WHh+vmm2+u03Xef//9+s1vfqPk5GR98cUXOnjwoH788Ue5urqqdevWGjZsWK2Ov0sZNGiQPvjgA/3tb3/TZ599pp9++klNmzbVTTfdpN/85jeaNGlSlQ+JvpT6OIZ79uypDRs26PXXX9e///1vHThwQKWlpWrdurUCAwM1derUSr9G+9WvfqWkpCStXr1aKSkpys7O1i+//KJWrVpp5MiR+t3vflfp4d4PP/yw/P399Y9//EPfffed9u7dK3d3d910000KCQnR5MmTjTvhLld9bf/auvHGG7Vw4UI9++yz2rp1q06fPq1u3brpt7/9re69994qHyh+KbXZb/fee686duyoVatWKTMzU5mZmWrevLluueUW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hm4AAAAAAAAA0NBsNpsWL16srVu3ymazKT09Xa+99prCwsIkSaGhodqwYYO2bdum0tJSJSQkKDc3VyEhIZKkcePGaeXKlcrOzlZhYaEWLVqkwMBAtW/fXh06dFDfvn21aNEiFRYWKjs7WytWrFBoaKgkKSQkRDk5OUpISFBpaam2bdumDRs2VHqmIAAAwKXwiz8AAAAAAAA0eiEhIZo9e7bmz5+vo0ePytfXV4888ojGjBkjSQoODta8efOMcn9/f61atUo+Pj6SpOjoaJWVlSk8PFxFRUUKCgrS8uXLjeXHx8drwYIFGjZsmFxdXTV27FhFRUVJkiwWi9asWaO4uDjFx8erZcuWio2NVf/+/a/0ZgAAAP/jSPwBAAAAAACgUdqzZ4/D+0mTJjkM9XmhMWPGGInACzVt2lQxMTGKiYmpstzX11fx8fHVLrtnz556++23a9BqAACA6jHUJwAAAAAAAAAAAGACJP4AAAAAAAAAAAAAE6izxF9hYWFdLQoAAKDRqU0sdeLECYWEhGj79u3GtHnz5qlHjx4KCAgwXu+8845RnpycrJCQEPXu3Vvjxo1Tenq6UVZeXq6lS5dqwIABCggIUGRkpI4dO2aU5+bmKioqSv369VNQUJDi4uJUVlZmlO/atUsTJkxQQECAhg4dqsTERKc/EwAAQG3QLwUAAHCO04m/wMDAKqffdtttTq+czioAANDY1FUstXPnToWFhengwYMO07/55hstXLhQ6enpxissLEyStH37di1cuFBLlixRWlqaRo8ercjISJ0+fVqStHLlSqWmpmrdunVKSUmRu7u7YmNjjWXPmDFDnp6eSklJUVJSkrZu3aqEhARJ0smTJzV9+nSNHTtWaWlpiouL0+LFi7V7926nPhcAAMDF1GW/FAAAgBk1qclMBw4c0B//+EfZbDYVFhbqvvvucygvLCzUtdde69SKd+7cqSeffLLazqrf/va3lerYO6tWrVqlXr16ae3atYqMjNTmzZvl4eHh0FnVokULzZ07V7GxsXrllVckneus8vPzU0pKinJychQZGamEhARFREQYnVWPPvqowsLClJaWpujoaHXt2lW9evVy6rMBAACcr65jqeTkZMXHx2vWrFmaOXOmMf3s2bPau3evevToUWW9xMREjRw5Un379pUk3X///XrnnXe0ceNGjR8/XomJiYqJiVHb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AAAAAAAAAMAESPwBAAAAAAAAAAAAJkDiDwAAAAAAAAAAADABEn8AAAAAAAAAAACACZD4AwAAAAAAAAAAAEyAxB8AAAAAAAAAAABgAiT+AAAAAAAAAAAAABMg8QcAAAAAAAAAAACYAIk/AAAAAAAAAAAAwARI/AEAAAAAAAAAAAAmQOIPAAAAAAAAAAAAMAESfwAAAAAAAAAAAIAJkPgDAAAAAAAAAAAATIDEHwAAAAAAAAAAAGACJP4AAAAAAAAAAAAAEyDxBwAAAAAAAAAAAJgAiT8AAAAAAAAAAADABEj8AQAAAAAAAAAAACZA4g8AAAAAAAAAAAAwARJ/AAAAAAAAAAAAgAmQ+AMAAAAAAAAAAABMgMQfAAAAAAAAAAAAYAIk/gAAAAAAAAAAAAATIPEHAAAAAAAAAAAAmACJPwAAAAAAAAAAAMAESPwBAAAAAAAAAAAAJkDiDwAAAAAAAAAAADABEn8AAAAAAAAAAACACZD4AwAAAAAAAAAAAEyAxB8AAAAAAAAAAABgAiT+AAAAAAAAAAAAABMg8QcAAAAAAAAAAACYAIk/AAAAAAAAAAAAwARI/AEAAAAAAAAAAAAmQOIPAAAAAAAAAAAAMAESfwAAAAAAAAAAAIAJkPgDAAAAAAAAAAAATIDEHwAAAAAAAAAAAGACJP4AAAAAAAAAAAAAEyDxBwAAAAAAAAAAAJgAiT8AAAAAAAAAAADABEj8AQAAAAAAAAAAACZA4g8AAAAAAAAAAAAwARJ/AAAAAAAAAAAAgAmQ+AMAAAAAAAAAAABMgMQfAAAAAAAAAAAAYAIk/gAAAAAAAAAAAAATIPEHAAAAAAAAAAAAmACJPwAAAAAAAAAAAMAESPwBAAAAAAAAAAAAJkDiDwAAAAAAAAAAADABEn8AAAAAAAAAAACACZD4AwAAAAAAAAAAAEyAxB8AAAAAAAAAAABgAiT+AAAAAAAAAAAAABMg8QcAAAAAAAAAAACYAIk/AAAAAAAAAAAAwARI/AEAAAAAAAAAAAAmQOIPAAAAAAAAAAAAMAESfwAAAAAAAAAAAIAJkPgDAAAAAAAAAAAATIDEHwAAAAAAAAAAAGACJP4AAAAAAAAAAAAAEyDxBwAAAAAAAAAAAJgAiT8AAAAAAAAAAADABEj8AQAAAAAAAAAAACZA4g8AAAAAAAAAAAAwARJ/AAAAAAAA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9JLy8/NltVqNnqqgoED5+fny8vIyArurjwHVqVPnNyMyZWRklAsHr2fTpk368MMP9dFHH6l+/fqSpNdee+03w7IDuP8w1CeAO9K6dWs98sgjeuWVV3TkyBEVFxfLarVq/fr1OnnyZLlhmK7lylVpb731li5evKicnBy9/vrrCgwMLBdsXc3Z2fmmV6hdERISYgzTWVRUpPPnz2vKlCkaP3687OzstGrVKk2ePFl5eXlydXWVq6urqlevLicnJ7333nt67bXXZLVa5ebmpmrVqpU7o/x6Hn30UXXs2FHTp09XTk6OLl68qLfeekulpaUKCQm56euvJScnRwsWLFBRUZEOHjyoVatWaeDAgddctkWLFqpfv75mzJihwsJCXbx4UfHx8Ro6dKhKS0uNs7ccHR1VUlKidevWaffu3SouLpZ0ef9KMg7IAQCAB9vN+imr1SpnZ2fZ29vr0qVLSkxM1NGjR43eol+/flq8eLGOHz+uS5cuacaMGcZVeYWFhSotLVW1atUkST/88INxX5mioiJ16tRJJSUleu+991RUVKT09HTFx8fr0qVLN627f//++uCDD4x73Pzf//2f+vbtqw0bNlTEbgIAALihqVOn6pNPPtG7776r/Px82Ww2ff/99xo5cqQef/xxRUREqHHjxoqJiVFBQYEKCgoUGxurevXqqVWrVtdcZ79+/bRw4UIdOnRIpaWl2rRpk7Zv364+ffrctJ6CggJZLBa5uLjIZrNp165dWrt2rdHDAbh/EfwBuCMuLi5asWKFvLy89MILL8jf31+dOnXS+vXrtXTpUjVq1OiGr69Ro4aWLl2qo0ePqmPHjurevbvq1Kmj2bNnX/c1ISEhOnfunPz8/HT+/Pkbrt/V1VXLli1TamqqOnTooC5dushisRj38ZswYYLq16+vsLAwtWrVSqtXr9b8+fPl7OysadOmqaysTJ07d1br1q319ddf37Cuq73xxhuqW7eu+vTpo3bt2unIkSN6//33yw2ZeTuaNWumU6dOKSAgQFOmTNHkyZONs+J/zcHBQQkJCcrNzVW3bt0UHBys9PR0LV26VM7OzoqIiJCPj49CQkLUvn17rV+/XoMGDTLO6GratKn8/PzUvn37331PQgAAYB4366fGjRunixcvql27dgoNDdVXX32lXr16Gb3FsGHD1LNnTw0ePFjBwcGqUaOG0RM1bNhQkydP1qRJk+Tn56exY8fq2WeflaOjo44ePSo3NzctWbJEe/fuVXBwsIYMGaKBAwdqwIABN637j3/8oyZMmKBXX31VrVq10tixYzV06FANGTKkwvYVAADA9bRp00YffvihvvvuO+M41JgxYxQYGKjFixfL0dFRCQkJKikp0VNPPaWQkBAVFxdr6dKl1x1+8/nnn9ef/vQnjR8/Xv7+/kpISNDMmTPVpk2bm9Zz5ZhVWFiYAgMDtWDBAj333HM6ceKEioqK7vbbB3AP2dlu5ZIZAAAAAAAAAAAAAFUaV/wBAAAAAAAAAAAAJnDta4QBANf14osvas+ePdedHxcXp549e97Sur755hs999xz151fu3Ztbdy48bZrBAAAAAAAAAA8eBjqEwAAAAAAAAAAADABhvoEAAAAAAAAAAAATIDgDwAAAAAAAAAAADABgj8AAAAAAAAAAADABAj+AAAAAAAAAAAAABMg+AMAAAAAAAAAAABMgOAPAAAAAAAAAAAAMAGCPwAAAAAAAAAAAMAECP4AAAAAAAAAAAAAEyD4AwAAAAAAAAAAAEzg/wHZHd8nRffsvwAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 1800x600 with 3 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "cols = ['Shortness_of_breath','Headache',\t'Corona']\n",
    "\n",
    "# Set up the figure and axes\n",
    "\n",
    "fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))\n",
    "\n",
    "# Plot univariate distributions for each column\n",
    "for i, col in enumerate(cols):\n",
    "    sns.countplot(data=df , x=col, ax=axes[i])\n",
    "    axes[i].set_title(f'Distribution of {col}' , fontsize=18 )\n",
    "    axes[i].set_ylabel('Count')\n",
    "    axes[i].tick_params(axis='x')\n",
    "    axes[i].grid(True)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 119,
   "id": "2f371eae",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABv4AAAJOCAYAAAB/dnBOAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAACLcUlEQVR4nOzde5xVdb0//tcMYAyIzOB40DqoBQymYiLGRT1aGlleOYjaiUwstYA66TeoTEqPBGhXzmSSWUYm5QmNCjMvlSWSXFQSjwUCGlIkAgLCIMpl//7gN/s4AsIoMMP2+Xw89kP3eq/LZ+3Ze20++7XWZ5UVCoVCAAAAAAAAgL1aeVM3AAAAAAAAAHjjBH8AAAAAAABQAgR/AAAAAAAAUAIEfwAAAAAAAFACBH8AAAAAAABQAgR/AAAAAAAAUAIEfwAAAAAAAFACBH8AAAAAAABQAgR/AAAAAAAAUAIEfwBvYn//+9/TrVu3bT4OP/zw9OjRI+9///tz2WWX5fe///121zNjxozichs3bnzD7SoUClm4cGGjl7vgggvSrVu3fOtb3ypOe+U+Llq06A23bWe9+OKL+fvf/95g2s9//vN069YtJ5544h5rx570xz/+Meeff3569OiRHj165IwzzkihUNipZVesWJHrr78+5513Xvr06ZMjjzwyxx9/fM4///zU1tZm6dKlu7n1AACUIn2e3UefZ+f6PCeffHK6deuWb3/7268539y5c9O7d+9069Ytxx9/fObPn78rm04zsWrVqixbtqxJ2/B6jz/A3kPwB0CSpKamJsccc0zx0b179xx00EH55z//md/85jcZMmRIPv7xj2fNmjW7tR1z5szJeeedl/Hjx+/W7exOU6ZMyamnnpqHHnqoqZuyx8yZMyef/OQn8+c//zn77LNP3vGOd+Tggw9OWVnZDpf9/e9/n1NPPTXf/va385e//CXV1dU54ogjst9+++Xxxx/Pd77znZx66qn5+c9/vgf2BACAUqXPs+vo8zSuz7Mjc+fOzYUXXphVq1blX/7lX/LjH/84Xbt23QWtpjmZMGFC3v/+9zdpqFsKxx9gx1o2dQMAaB5GjhyZ3r17bzX95Zdfzi9/+cuMHTs2Dz74YIYOHZof/OAH2WeffYrzHHXUUbnrrruSJC1bvrGvlp/85CeZM2dODjnkkEYve9111+XFF19MVVXVG2rDG/Wtb31rm1eo9evXL+9617vSqlWrJmjV7nXPPfdk8+bN+dd//df8+te/TuvWrXdquYULF+ayyy7LSy+9lE996lP52Mc+lrZt2xbrzz//fL7xjW/k9ttvz5VXXpkDDzwwxx133O7aDQAASpg+z66jz7PzfZ4dmTt3bgYPHpxVq1bloIMOyo9+9KPX9d6g+Rs7dmxTN+ENHX+AvYcr/gB4Tfvss0/OPffc3HjjjWnRokVmzpyZH/3oRw3mqaioSOfOndO5c+cmauUWb33rW9O5c+d06NChSduxPe3atUvnzp1z8MEHN3VTdrmVK1cmSd71rnc1qgN8880356WXXsppp52WT3/60w1CvyTp0KFDRo0alb59+2bz5s25/vrrd2m7AQBAn2fX0edpnHnz5mXw4MFZuXJl/vVf/zW33nqrQAaAN0zwB8BOefe7353zzz8/SfL9738/69evb+IW0Zxs3rw5SRqcFb0zHn/88SRbOs/bU15enoEDBxbnr98WAADsSvo8vJbX2+fZnnnz5uXCCy/MypUrc8ghh+TWW2/Nv/7rv+6SdQPw5ib4A2Cn1XeCV61alYcffrg4/bVudP/YY4/l8ssvz6mnnpqjjjoq7373u3Peeefle9/7XtauXbvVOiZPnpxkyz0junXrlgsuuCDJ/92w/vjjj8/SpUtz6aWX5qijjkqvXr3y2c9+Nsm2b3T/Sps3b86Pf/zjnHHGGenevXuOO+64fOYznymGT6/0WvtUr74+Y8aMJMm3v/3tdOvWLf/4xz+SbBlK6JU3cd/Rje4feuihfPrTn84JJ5yQI488Mn369MnFF1+ce++9d5vz198kfuHChZk5c2YuvfTS9O7dO927d88HP/jB1NbWpq6ubpvLvpZNmzZl0qRJueCCC/Lud787Rx55ZN7znvdkxIgReeKJJxrMW7/P9X+3yZMnF1+Xv//97zvcVv0QQH/4wx9SKBS2O9/JJ5+cX/7yl7nvvvtSXr71P1+WL1+er371qznttNPyrne9Kz169Mg555xTvKLwlWbMmJF3vvOd6dat2zbva7Bw4cK8613vSrdu3fLd7353h/sAAEDp0OdpSJ/njfd5tuXJJ58sXun3jne8I7feemsOOuigbc5b/5pefvnlWbduXcaNG5dTTz013bt3T+/evfPJT36ywXv11Z5++ulcddVV6devX4488sj07Nkz5513XiZMmNAg3K6rq0v37t3TrVu3PProo1utZ/z48enWrVsOP/zwbd4H81Of+lS6deuWm2++OUnyhS98Id26dctPf/rT/P3vf88VV1yRE088MUceeWROPPHEXHnlla/79XstDz/8cD772c/mve99b/E99slPfnK796NcvXp1rr/++vTv3z89evTIu971rnzwgx/Mddddl+eee26r+V/P36P+tah30UUXpVu3bg3uY18oFPK73/0u//mf/5n3vve9Oeqoo3LUUUfl5JNPzogRI7b5Ga43d+7cfOlLX8r73ve+dO/ePe9+97tz4YUX5u677y7Os6PjD1BaBH8A7LTDDjss++67b5Jk5syZO5z/3nvvzYc//OHcddddWb16dbp06ZIOHTpkzpw5+cY3vpEPfehDxY5wu3btcswxx2T//fdPsmWIx2OOOSY1NTUN1vnyyy/n4x//eP70pz+lc+fOKSsry9ve9radav+XvvSlfOUrX8myZctSU1OTl19+OXfffXfOO++83HHHHY15KbbpoIMOyjHHHFM8A/SQQw7JMcccs90O3CuNGjUqgwcPzr333psNGzbksMMOS6tWrTJ16tR8+tOfzmWXXZYNGzZsc9lJkyblox/9aKZPn56OHTumQ4cOeeqpp/Kd73wnH//4x7Np06ad3oe1a9dm0KBBGTlyZGbOnJl27dqlW7duWbNmTX71q19l4MCB+eEPf7jVPtf/3fbff/8cc8wxOeaYY/KWt7xlh9s74YQTkmz5AeCCCy7IPffck3Xr1m01X5s2bXLYYYflwAMP3Kr2yCOP5PTTT88PfvCDPPPMM+nUqVPe+ta35oknnsh1112X8847L8uWLSvO37t37wwePDhJ8p3vfCcLFiwo1l5++eV89rOfzfr169OnT59ceumlO/fCAQBQEvR5Xps+T+P7PK82f/78XHjhhXn++edTU1OTW2+9Nf/yL/+yw+VeeOGFnH/++Rk/fnzWrVuXLl26ZN26dbn//vvz0Y9+NH/4wx+2WuZXv/pVzjrrrNx222157rnnUlNTk+rq6jz22GMZO3Zszj333Dz77LNJkrZt2xbvgfnggw9uta4//elPSbaEpq/+bLz88suZNm1akuSUU05pUPvLX/6Ss88+O7/4xS9SUVGRQw45JEuXLs3tt9+ec889N//85z93/KLtpG9+85v5yEc+kjvvvDPr1q1Lt27dUl5envvvvz+DBw/O//zP/zSYf968eTnjjDPy7W9/O/Pmzcvb3va2vP3tb88zzzyTm2++OWeccUYx9H61xvw9Dj300BxzzDHF5zU1NQ3eU4VCIcOHD8/QoUNzzz33ZNOmTenatWsOOOCA/POf/8yvfvWrfOhDH8of//jHrdoxceLEDBw4MD/72c/y/PPPp2vXrmnTpk2mT5+ez3zmM8WTBHb2+AOUiAIAb1qLFy8u1NTUFGpqagrTp0/fqWXOPPPMQk1NTeGzn/1scdr06dOL69mwYUOhUCgUNm3aVDj++OMLNTU1hZtuuqmwcePG4vz/+7//W+jTp0+hpqamcOONNzZY/+c///mt1v/qtvbq1aswf/78QqFQKLz00kuFNWvWFAqFQuEjH/lIoaampvDNb35zm8vV1NQUvvWtbxVefvnlQqFQKKxfv75w9dVXF2pqagpHHHFEYcGCBa+5T6+2vdfuve99b6Gmpqbws5/9rMH0O+64o1BTU1P4t3/7twbTf/CDHxRqamoKhx9+eOHWW28tbNq0qVi76667CkcffXShpqamMGrUqG1up6ampvCFL3yh8MILLxQKhUJh8+bNhVtvvbVYu++++7bZ/m35xCc+UaipqSkcf/zxDfbrpZdeKowbN664znvuuafBcvV/t89//vM7va1CoVB44YUXCqeddlqDv9ERRxxR+NCHPlT4+te/XvjjH/9YWLdu3XaXf/bZZwu9evUq1NTUFEaOHFlYvXp1sbZo0aLCueeeW6ipqSl8+MMfbrDcSy+9VDjrrLMKNTU1hXPPPbf4/hw7dmyhpqam0KdPn8LSpUsbtS8AADQ/+jz6PK+2p/s89ftQW1tbePLJJwt9+/YtbuPee+/d4fL1r2l9m6dOnVqsLV26tPh+Peussxos9+c//7lw+OGHF/tK9e+hQqFQ+Mtf/lJ4//vfX6ipqSn8+7//e/HvP3HixEJNTU3hvPPOa7Cuurq6whFHHFFsx1e+8pUG9QcffLBQU1NTOP3004vT6l+v+vU99dRTxdqjjz5a6NGjxzbX9XrdeeedhZqamsI73/nOwoQJE4qfx40bNxZuuOGGYq3+M7BmzZrCCSecUKipqSmcf/75hWeeeaa4rmXLlhXfJz179mxQe71/j0Lh/z5P06ZNazC9fp1HHXVU4Q9/+EOD2oIFCwqnn356oaampjBgwIAGtUceeaRw2GGHFWpqagpf//rXC+vXr2+wzvraK9u4veMPUFpc8QdAo7Rt2zbJlqFvXsvzzz9fvMrqvPPOS4sWLYq1I444Ipdffnne9773pbKystFt+PCHP5wuXbok2XJ/hfozcnfkjDPOyGWXXVYcXvItb3lLvvzlL6dnz57ZsGFDcUiSPemll14qDjf5n//5nxk0aFCDoSw/+MEP5itf+UqS5Cc/+ck2h0I57LDDMmbMmLRr1y5JUlZWlkGDBhWHEnnkkUd2qi1//vOfc//99ydJamtri2d7Jlte58985jPFoY++/vWvN3ZXt6ldu3b5yU9+kgEDBhTfIxs2bMijjz6a733ve7nkkkvSp0+fjBgxIkuWLNlq+R/84AdZtWpVTj755IwaNSr77bdfsXbwwQfnhhtuyL777puHH364wdmR++yzT77+9a/nLW95Sx577LH8+Mc/zkMPPZQJEyakrKwsY8eO3amzbgEAKD36PLvWm73PU2/BggW58MILs2LFiuL+jxo1KitXrtzpdXz5y18ujpqSJP/yL/+ST33qU0m2DPf4ymFPa2trs3HjxpxwwgkZNWpUg/fQO9/5znz/+99P69at88QTT+TXv/51kv+7Wu/xxx/P6tWri/PPnDkzGzZsyLvf/e4k2eoquPrX9NVX+yVbbu9w/fXX5+1vf3txWo8ePTJgwIAk2eawoq/H9ddfn2TLMJoXXnhh8fPYokWLDBkyJMcff3w2bdqUX/ziF0m2vNeee+65VFdX58Ybb0ynTp2K66qurk5tbW1qamqyZs2a7d4CojF/j9cybdq0tGzZMh/+8Idz0kknNah17tw5F198cZItQ8S+0g033JDNmzfngx/8YD772c82uAJ1wIABGThwYJI0GFIUeHMQ/AHQKPVDr5SVlb3mfFVVVWnfvn2SZPjw4Zk9e3bxZujJlo7xd77znZx33nmNbkPPnj0bvUySfOQjH9lqWllZWbEN2xo2Y3d7+OGH88ILL6Rly5YZNGjQNuc57bTT0rFjx2zatGmbw7e85z3v2ebf4x3veEeSbPP+C9tS31k76qijGgxD8kof+9jHkiSLFi3aqtPxerVv3z5jx47N/fffn5EjR+akk05q0Cldv359fvWrX+W0004rDi9T77e//W2S5Kyzztrmuqurq3P88ccn+b/9q9e1a9fivVJqa2vz+c9/PoVCIR/96Efznve8Z5fsGwAAex99nl1Ln2eLu+++OytWrEjPnj3z05/+NG95y1uydOnSXHHFFTu1fIsWLbZ578TOnTsX/79+WNl169YVw7mPfvSj21xfp06d8r73vS9J8rvf/S5J0rFjxxxxxBHZtGlTg3vi1Q/j+ZGPfCT77bdfnnzyyTz//PPFev37alvB35FHHpkDDjhgq+mN/du9lkWLFuWpp55KknzoQx/a5jyjR4/Ob3/721x++eVJkt///vdJkv79+xc/x6+0zz77FO9/9/vf/36re9I35u+xI9/4xjcyZ86cYtteraKiIsmWIVXrjzEvvvhipk+fnuT/7k36apdddlnuvvvuXHfddTvVDqB0tGzqBgCwd6n/R/krr6zalhYtWmT48OH50pe+lD/+8Y/54x//mPbt26d37945/vjj8573vGeb92vbGdvqNOyMI444YpvT688SXbZsWV544YUd7tuuVN85OeSQQ7Z7Fm9ZWVkOP/zwLF26NE8//fRW9e1dmda6desk2en7XdS3ZXuvU7Ll3gT77rtv1q5dm6effnqX3g+gY8eOueCCC3LBBRdk06ZN+ctf/pI//elPufPOO/Pkk0/mxRdfzLBhw/K73/0uHTp0SF1dXf7xj38k2XKm4y233LLN9dbPU79/r/TRj340DzzwQB588MHU1dXliCOOyPDhw3fZPgEAsPfR59m19Hn+T69evXLjjTemTZs2+X//7/8VT4D80Y9+lAsvvPA1l23fvn1xf1/plVd5bdy4MUmyePHiYoB95JFHbnedRx55ZO68884Gr/nJJ5+cJ554ItOmTcsHPvCBJFuCv/Ly8vTp0yfHHHNM/vCHP2TGjBn54Ac/mIULF+aZZ55Jx44d071796220bFjx21uu35f6tv8RixatCjJlnvDv/LKvVd69X0o6/f5td4L9bXnn38+q1atSlVVVbHWmL/HzmjRokVefvnlPPTQQ3nqqaeyePHi/O1vf8vcuXMb3Adx8+bNKS8vz5IlS4p/48MOO2yb69x///2L9/QD3lxc8QfATnv55ZeLIcorz2LbnvPOOy+33HJL3vve9+Ytb3lLVq9enXvvvTdXXXVV3vOe9+STn/xk8UbijbGtf1zvSKtWrYo3oH+1+qF8ki1nze1J9WcA1g9Zsz31HeRtDRWyvf2q9+ozE99oW+pfr50dtuT1aNGiRbp3755PfOIT+dWvfpUvfvGLSbacufrLX/6yQXuTLUOePProo9t8LF26NMm2zyQtKyvLe9/73uLzt7/97Tt8PQEAKF36PLuePs8Wxx57bG666aa0adMmSXLhhRemb9++SZKvfe1r+ctf/vKay9cP3/pa6l+HV/aVXmtft/Wa11+19+CDDyZJli5dmoULF+ad73xnKisri22uv9qs/grNU045ZZtXZe5Mu9+o+mF5X/k+35GdeS+8Mqh+9XuhMX+PHdmwYUPGjRuX97znPRk8eHCuueaa/PCHP8wDDzyQysrKnHrqqVst88qhiBuz38Cbgyv+ANhpc+bMKZ5Rtr1hUV6td+/e6d27d9avX5+HH344s2bNytSpU/PEE0/k/vvvzz//+c/84he/2OEwOm/Uhg0b8vLLL2+zw/jKQGhbZ75u6x/r69at2yXtqv8H+o6GN3nhhRcazL877Gxb6utvtC0/+9nPMmHChHTq1Ck33njjducrKyvLhRdemN/85jeZPXt28czM+uFOkmTKlCmv60zcxYsX55vf/GaSpLy8PHfeeWfe97735YMf/GCj1wUAwN5Pn+f/6PPs2rb06dOnQaBbVlaWa6+9NmeddVZWr16dyy+/PD//+c93yTZfuY41a9Zs96qv+vv4vXL+d77znXnrW9+aJUuW5Kmnnsqf//znJCkGfvX/rR9K9JXBX1OpD1MbE9S2bds2q1evfs33wivvc7g735df/vKX8/Of/zwtWrTI+eefn3e/+93p2rVrDj300LRu3TrTpk3LPffc02CZ+n1OtoSYHTp02G3tA/Y+rvgDYKdNmjQpyZZhVupv6r09L7/8chYuXJjHHnssyZYzVk844YRiZ6Y+bJk7d27mzZu3exv+/9vWUI9JimdWHnzwwcUwqf5G4MmWfXm15557bpe0qf6+BosWLdru+P+bN28utvGQQw7ZJdt9rbY88cQT251n4cKFxR8A3mhbWrRokYULF+ahhx7aqRva1w93VD+8yn777Zfq6uokyYIFC7a73Lx58/LXv/61Qact2TIc0IgRI1JXV5eTTjqpeD+Fq6++uniVIAAAby76PP9Hn2f3tiVJDjzwwFx11VVJkr/97W+55pprdsl6Dz744OIVaf/7v/+73fnqa6/ez/pRUR588MFiwHfccccl2TJsbHV1dZ5++uksWLAgjz76aNq1a5fevXvvkra/HoceemiSLWH13//+923O87vf/S4XXHBBvvrVrybZufdC/evTvn37BsN87kpLly7N5MmTkySjRo3KNddckzPPPDOHHXZYMSje1lXDnTp1Kn6G58+fv811P/744/mP//iPXHHFFTt99SFQGgR/AOyUmTNn5le/+lWS5NJLL23QSdyWBx54IKeddlouvfTSbXYi6zsNScP7MdSfBbs7/lF6xx13bDVt06ZN+elPf5pky70M6r3yH/Xb6jzfd999291OY/ahZ8+ead++fTZu3JiJEyduc55f//rXWbZsWcrKyvJv//ZvO1zn61XfuZszZ04effTRbc4zYcKEJFs6qPX3CXm9TjnllLRt2zYvvfRSsfO1PStWrCjeXP4973lPcXr9/996663Fm5y/0po1a3LhhRemf//++dGPftSgduONN2b27Nlp165dRo0alY9//OM54ogjsmrVqnzhC1/QMQIAeJPR52lIn+eN93l2xumnn54zzjgjSfKLX/yieGuDN6JNmzbFIG5790JfvHhxfv/73ydJTjzxxAa1+vfJtGnTMnPmzLRq1So9e/Ys1vv06ZNkyxClGzduzEknnbRHhvTcns6dO+dtb3tbkm1/BpJk8uTJmTlzZp5//vkk//de+MUvfrHVSaLJljC8/nOzq96T2/rcLFmypPh8W/cb3Lx5c37+858Xn9cfS/bdd9/i32R7+3znnXfm0Ucfzd///vfitnfn8QdoPgR/ALymurq6TJw4MZ/4xCeyefPm9O3bN//xH/+xw+VOPPHEVFVVZdWqVfn85z/fYPz5urq6XHfddUm23GC7a9euxVr98BlLlizZtTuS5Mc//nEmTpxYDIjWrl2bz33uc3niiSfSvn37fOxjHyvO+/a3v704HMpXv/rV4rAzhUIhU6ZMyXe+853tbqd+yI36e4O8loqKilx66aVJktra2gbtS5J77rknX/7yl5NsuX/I29/+9sbscqP06NEjJ510UpLkP//zP4tndiZbOj21tbX52c9+liT53Oc+94aHKqqsrMz/+3//L0ny85//PBdffHHmzJnToAOyefPmPPTQQ/noRz+aNWvW5LTTTkuPHj2K9UsvvTRt2rTJI488khEjRhQ7ccmW1//SSy/NypUr065duwwaNKhYe/zxx3PDDTckSa644op07NgxLVq0yNixY9OqVav86U9/2m4HGQCA0qLPo8+T7J4+z8666qqrctBBByVJ/uu//iuLFi16w+v81Kc+lZYtW+bBBx/Ml770pQZXW86dOzeXXHJJXnrppRx22GHp379/g2V79eqVfffdN9OmTcuSJUvSo0ePBsOU1g/32RyG+Uy2hFlDhw5Nktx0002ZNGlSsV+5adOmfO9738t9992Xli1bZvDgwUmS//iP/0jHjh2zfPnyfOITn8jixYuL61uxYkU+85nP5Mknn0zbtm3z6U9/epe0s/5z88rP/iGHHFI8yeCmm25qcA/OJUuW5DOf+Uwefvjh4rRX1ocOHZqysrL88pe/zHe/+91s3LixWPvFL36RH//4x0mSiy++uDh9dx5/gObDPf4ASJJ85StfaXDj6o0bN+aFF17I4sWLi2eUnXzyyfna176Wli13/PWxzz775L//+7/z8Y9/PHfddVd+97vf5eCDD055eXkWL16cdevWpaKiItdee22De1C8853vTJI8+uij+cAHPpAuXbrk+uuvf8P716pVq5xwwgm55pprMn78+HTs2DFPPfVU1q1bl7Zt2+a///u/07Fjx+L85eXlueyyy/KlL30pM2fOzEknnZS3v/3tee6557Js2bK8973vzfPPP18c1ueVDj/88Dz55JP5/ve/nwceeCD9+vUrdkK25eMf/3j+/ve/56c//WmuueaafPvb306nTp3y7LPPFofXOfXUU3PllVe+4ddhR7761a/mk5/8ZGbPnp2PfvSjedvb3pYOHTrk6aefztq1a9OiRYtcdtllOf3003fJ9j7ykY9k48aNGTduXKZOnZqpU6emqqqq2On9xz/+UTz78gMf+ECuvfbaBssfcsghGTduXC6//PLceeedueeee9KlS5ds2LAhf/vb37Jx48a0adMm3/ve94o/arz44osZMWJENmzYkBNPPDHnnHNOcX3dunXLJz/5yXz729/ON77xjRx33HENfqQBAGDvpc+jz5Ps+T7Pzthvv/0yduzYXHTRRamrq8vll1+e2267bZv3a9xZPXr0yOjRozNy5Mj87Gc/y69+9at07tw569atK943vaamJtdff/1W29lnn31ywgkn5O67707yf0FfvVdezdqqVautrhhsCgMHDsyCBQvywx/+MCNHjsy4ceNy4IEH5u9//3tWrVqVFi1a5Oqrr85hhx2WZMtr/t3vfjeXXnppZs+enfe///3p0qVLWrZsmfnz52fDhg2prKzM17/+9eJQom/U4YcfnlmzZuWaa67JT3/603z4wx/OwIEDc9FFF+X73/9+7rzzzvzxj3/MwQcfnLq6uixatCiFQiG9e/fOI488ko0bN+bZZ59NZWVlki1/lyuuuCLXXnttvvWtb+Xmm28ufq6WL1+eJBk2bFgx7E523/EHaF4EfwAkSZ588skGz8vLy1NRUZF3vOMd6d69e84666yt/rG/I717986kSZPywx/+MI888kj+9re/pWXLljnwwANzwgkn5GMf+1je+ta3Nlimf//+eeqpp/LLX/4yf//737Nhw4ZtDuHYWGVlZfn2t7+dm266KZMnT868efNSVVWV0047LZ/85CfTqVOnrZY577zzcuCBB2bChAmZM2dOFi5cmLe//e35xCc+kUGDBjW4guyVPv/5z+fFF1/Mn/70pzz11FNZuHDhDtt29dVX533ve19++tOf5s9//nP++te/pqqqKu9973szcODAvO9973vDr8HOqKyszI9//ONMnjw5v/rVrzJv3rwsW7YsHTt2zAc+8IEMGjQohx9++C7d5uDBg/OBD3wgkydPzp/+9Kc888wzeeqpp1JeXp4DDjggp5xyymu+/0466aT8+te/zoQJEzJ16tQ8/fTT2bRpU972trfl+OOPz8c+9rEGf9/rrrsuTz/9dHGIz1f7xCc+kd/+9rf561//mhEjRuRnP/vZG+pwAwDQPOjz6PMkTdPn2Rl9+/bN4MGD88Mf/jBPPPFEvvGNb+SKK654Q+vs379/unfvngkTJuRPf/pT5s+fnzZt2uSYY47JGWeckYEDB+Ytb3nLNpc95ZRTthv8vfWtb82hhx6av/3tb+nTp0+DQL0pfeELX8iJJ56YW2+9NX/+858zd+7ctG/fPh/4wAdy8cUXp3v37g3mP/zww3PnnXfmRz/6UX73u9/lmWeeSVlZWd7+9rfn5JNPzoc//OEGYfkbNWbMmFx99dWZPXt2nn766fztb39LkowYMSLdu3fPLbfckqeffjrz5s1L+/bt07dv3/z7v/97zjzzzFxwwQWZNWtW7r///mJ4mSQXXnhhevTokR/96EeZOXNm5s2bl7Zt2+akk07K4MGDG4S0yfaPP+XlBgaEUlJWMKAvAAAAAAAA7PVE+QAAAAAAAFACBH8AAAAAAABQAtzjDwAAAACAJjNq1Kj85S9/afRyBxxwQGpra3dDiwD2XoI/AAAAAACazJNPPplHH3200cu97W1v2w2tAdi7lRUKhUJTNwIAAAAAAAB4Y9zjDwAAgF1m7ty5ueiii9KrV68cf/zx+dznPpfnn38+SXLVVVflyCOPTI8ePYqP//mf/ykuO3ny5PTr1y9HH310BgwYkNmzZxdrmzZtynXXXZfjjjsuPXr0yJAhQ/Lcc88V6ytWrMjQoUNz7LHHpnfv3hk9enQ2btxYrD/22GM599xz06NHj5x88smZNGnSHng1AAAA9izBHwAAALvE+vXrc/HFF6dHjx558MEHc+edd2bVqlX54he/mCR5/PHHM2rUqMyePbv4OP/885MkM2bMyKhRo3Lttddm1qxZOeusszJkyJC8+OKLSZLx48dn2rRpueOOOzJ16tS0bt06I0eOLG77sssuS5s2bTJ16tTcfvvteeihhzJhwoQkyerVq3PppZemf//+mTVrVkaPHp2xY8dmzpw5e/YFAgAA2M0EfwAAAOwSS5YsyWGHHZZhw4Zln332SVVVVc4///zMmjUrL7/8cp588skceeSR21x20qRJOf3009OzZ8+0atUqgwcPTlVVVe66665i/ZJLLslBBx2UfffdN1deeWUeeOCBLF68OIsWLcrMmTMzYsSIVFRUpFOnThk6dGgmTpyYJLn33ntTWVmZQYMGpWXLlunbt2/OPPPMYh0AAKBUtGzqBpSiFSvWxJ0TYWtlZcn++7fzGQHgdfE9Aq+t/jPSlN7xjnfk+9//foNp99xzT4444ojMnTs3GzduTG1tbR555JG0a9cu55xzTi6++OKUl5dnwYIFOeeccxos26VLl8ydOzdr1qzJs88+m5qammKturo67du3z7x585IklZWV6dixY7HeuXPnLFmyJC+88ELmz5/fYNn6dd9+++2N3kfHIF7JdxOwsxwvgJ3leMG2NKa/J/jbDQqF+EDCa/AZAeCN8D0Ce4dCoZBx48bl/vvvz6233prly5enV69eueCCC/LNb34zf/3rXzNs2LCUl5fn4osvTl1dXSoqKhqso3Xr1lm3bl3q6uqSJG3atNmqXl979bL1z+uX3966AQAASongDwAAgF1q7dq1ueKKK/LEE0/k1ltvTbdu3dKtW7ccf/zxxXmOOuqoXHjhhbnrrrty8cUXp6KiIuvXr2+wnvXr16eqqqoY2tXf7++V9bZt26ZQKGxVq3/etm3bVFRUZM2aNdtctrGa+qpKmifvC2BnOV4AO8vxgtdL8AcAAMAu88wzz+SSSy7JW9/61tx+++3p0KFDkuS3v/1tli9fng996EPFeV9++eW0bt06SdK1a9fMnz+/wboWLFiQE088Me3bt0/Hjh2zYMGC4pCdy5Yty6pVq1JTU5PNmzdn1apVWb58eaqrq5MkCxcuzIEHHph27dqlpqYm06ZN22rdXbt2bfT+GXKJVzIUF7CzHC+AneV4wbYY6hMAAIA9bvXq1bnwwgvTp0+fjB49OuXl5cVaoVDI2LFjc8ghh6RPnz7585//nFtuuSVXXHFFkmTgwIEZNmxYPvjBD6Znz56ZOHFiVqxYkX79+iVJBgwYkPHjx6d79+6pqqrKmDFj0qtXrxx88MFJkp49e2bMmDG55pprsnLlytxwww0ZOHBgkqRfv3752te+lgkTJmTQoEF55JFHMmXKlNxwww2N3kfDDbMt3hfAznK8AHaW4wWvV1mh4K2zqy1fLomHbSkrS6qr2/mMAPC6+B6B11b/GWlKP/zhD3PttdemoqIiZWVlDWqzZ8/Obbfdlh/+8IdZunRpqqurc9FFF2XQoEHFeX75y19m/PjxWbp0abp06ZKRI0fmXe96V5Jkw4YN+e///u/86le/Sl1dXXr37p1Ro0Zl//33T5IsX74811xzTWbMmJHy8vL0798/w4cPT4sWLZIkjz/+eEaPHp0nn3wyHTp0yNChQzNgwIBG76NjEK/kuwnYWY4XwM5yvGBbGtPfE/ztBj6QsG2+tAB4I3yPwGtrDsHfm4FjEK/kuwnYWY4XwM5yvGBbGtPfK9/xLAAAAAAAAEBzJ/gDAAAAAACAEiD4AwAAAAAAgBIg+AMAAAAAAIASIPgDAAAAAACAEiD4AwAAAAAAgBIg+AMAAAAAAIASIPgDAAAAAACAEiD4AwAAAAAAgBIg+AMAAAAAAIASIPgDAAAAAACAEiD4AwAAAAAAgBIg+AMAAAAAAIASIPgDAAAAAACAEiD4AwAAAAAAgBLQsqkbwJ5VXl6W8vKypm4Gb3ItWjjngKaxeXMhmzcXmroZAAAAAHuc34b3Ln5D3Ts0x98bBX9vIuXlZamsbOOAQZOrqmrb1E3gTWrTps1ZtWpds/syBgAAANidysvLUllVkRblLZq6Kewkv6HuHTZt3pRVK19sVr83Cv7eRMrLy9KiRXlG/mRqnn5udVM3B2CPevu/tM9XPvxvKS8va1ZfxAAAAAC7W3l5WVqUt8h/3fNf+dvKvzV1c6AkHFp1aK469apm93uj4O9N6OnnVmfuP55v6mYAAAAAALAH/W3l3/LksiebuhnAbmTMRwAAAAAAACgBrvgDAHaaG4HTHLhfMU2lOd60HQAAAF5J8AcA7JTy8rJUVVakvIUbgdO03OCcprJ506asXNW8btoOAAAAryT4AwB2Snl5WcpbtMjyn38hG5Y/1dTNAdijWlW/I9UDrm12N20HAACAVxL8AQCNsmH5U9nw7F+buhkAAAAAwKu4QQoAAAAAAACUAMEfAAAAAAAAlADBHwAAAAAAAJQAwR8AAAAAAACUAMEfAAAAAAAAlADBHwAAAAAAAJQAwR8AAAAAAACUAMEfAAAAAAAAlADBHwAAAAAAAJQAwR8AAAAAAACUAMEfAAAAAAAAlADBHwAAAAAAAJQAwR8AAAAAAACUAMEfAAAAAAAAlADBHwAAAAAAAJQAwR8AAAAAAACUAMEfAAAAAAAAlADBHwAAAAAAAJQAwR8AAAAAAACUgCYN/p5//vn069cvM2bMKE675557cvbZZ+eYY47JySefnOuvvz6bN28u1idPnpx+/frl6KOPzoABAzJ79uxibdOmTbnuuuty3HHHpUePHhkyZEiee+65Yn3FihUZOnRojj322PTu3TujR4/Oxo0bi/XHHnss5557bnr06JGTTz45kyZN2s2vAAAAAAAAAOwaTRb8PfLIIzn//PPzzDPPFKf97//+bz73uc/lsssuy8MPP5ybbropP//5zzNhwoQkyYwZMzJq1Khce+21mTVrVs4666wMGTIkL774YpJk/PjxmTZtWu64445MnTo1rVu3zsiRI4vrv+yyy9KmTZtMnTo1t99+ex566KHiulevXp1LL700/fv3z6xZszJ69OiMHTs2c+bM2WOvCQAAAAAAALxeTRL8TZ48OcOHD8/ll1/eYPo//vGPfOhDH8p73/velJeXp3PnzunXr19mzZqVJJk0aVJOP/309OzZM61atcrgwYNTVVWVu+66q1i/5JJLctBBB2XffffNlVdemQceeCCLFy/OokWLMnPmzIwYMSIVFRXp1KlThg4dmokTJyZJ7r333lRWVmbQoEFp2bJl+vbtmzPPPLNYBwAAAAAAgOasSYK/E044Iffdd19OO+20BtNPPfXUXHHFFcXn69evzx/+8IccccQRSZIFCxakpqamwTJdunTJ3Llzs2bNmjz77LMN6tXV1Wnfvn3mzZuX+fPnp7KyMh07dizWO3funCVLluSFF17I/Pnzt7tuAAAAAAAAaO5aNsVGDzjggB3Os3bt2nzmM59J69atM3jw4CRJXV1dKioqGszXunXrrFu3LnV1dUmSNm3abFWvr7162frn9ctvb92NVVbW6EUA2IMcpwF4I5rr90hzbRcAAAB7TpMEfzvy1FNP5T//8z+z//7755Zbbsm+++6bZEtQt379+gbzrl+/PlVVVcXQrv5+f6+st23bNoVCYata/fO2bdumoqIia9as2eayjbX//u0avQwAe0ZVVeOP6wBQz/cIAAAAzVmzC/7++Mc/5v/9v/+X8847L5/97GfTsuX/NbFr166ZP39+g/kXLFiQE088Me3bt0/Hjh0bDAe6bNmyrFq1KjU1Ndm8eXNWrVqV5cuXp7q6OkmycOHCHHjggWnXrl1qamoybdq0rdbdtWvXRu/DihVrUig0erHdrkWLcj9UAG96K1fWZdOmzU3djL2S7xGA5v09UlbmJEQAAIA3uya5x9/2/PnPf86wYcNyxRVX5POf/3yD0C9JBg4cmClTpmT69OnZsGFDJkyYkBUrVqRfv35JkgEDBmT8+PFZvHhx1q5dmzFjxqRXr145+OCDc+ihh6Znz54ZM2ZM1q5dm8WLF+eGG27IwIEDkyT9+vXL8uXLM2HChGzYsCHTp0/PlClTcs455zR6PwqF5vkAYIumPh7vrQ8Atmjq47FjNQAAANvTrK74++53v5uNGzdm9OjRGT16dHF6z5498/3vfz99+/bNVVddlauvvjpLly5Nly5dctNNN6WysjJJMmzYsGzcuDGDBg1KXV1devfunXHjxhXXU1tbm2uuuSannHJKysvL079//wwdOjRJUlVVlZtvvjmjR49ObW1tOnTokJEjR6ZPnz578iUAAAAAAACA16XJg7958+YV//+73/3uDuc/++yzc/bZZ2+z1qpVqwwfPjzDhw/fZr26ujq1tbXbXXf37t1z22237bANAAAAAAAA0Nw0q6E+AQAAAAAAgNdH8AcAAAAAAAAl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mTFjRnHaVVddlSOPPDI9evQoPv7nf/6nWJ88eXL69euXo48+OgMGDMjs2bOLtU2bNuW6667Lcccdlx49emTIkCF57rnnivUVK1Zk6NChOfbYY9O7d++MHj06GzduLNYfe+yxnHvuuenRo0dOPvnkTJo0aTe/AgAAAHuW4A8AAIBd7pFHHsn555+fZ555psH0xx9/PKNGjcrs2bOLj/PPPz9JMmPGjIwaNSrXXnttZs2albPOOitDhgzJiy++mCQZP358pk2bljvuuCNTp05N69atM3LkyOK6L7vssrRp0yZTp07N7bffnoceeigTJkxIkqxevTqXXnpp+vfvn1mzZmX06NEZO3Zs5syZs2deEAAAgD2gWQZ/TzzxRAYNGpRjjz02J5xwQr7yla/k5ZdfTrLjMzR359mhAAAA7NjkyZMzfPjwXH755Q2mv/zyy3nyySdz5JFHbnO5SZMm5fTTT0/Pnj3TqlWrDB48OFVVVbnrrruK9UsuuSQHHXRQ9t1331x55ZV54IEHsnjx4ixatCgzZ87MiBEjUlFRkU6dOmXo0KGZOHFikuTee+9NZWVlBg0alJYtW6Zv374588wzi3UAAIBS0OyCv82bN+cTn/hETj311MycOTO33357Hnzwwdx00007PENzd54dCgAAwM454YQTct999+W0005rMH3u3LnZuHFjamtrc9xxx+XUU0/N9773vWzevDlJsmDBgtTU1DRYpkuXLpk7d27WrFmTZ599tkG9uro67du3z7x58zJ//vxUVlamY8eOxXrnzp2zZMmSvPDCC5k/f/52190YZWUeHg0f3hceHh47+3C88GjKB7B7NafPccvdt5uvz+rVq7Ns2bJs3rw5hUIhSVJeXp6KiooGZ2gmaXCG5lFHHdXg7NAkGTx4cP7nf/4nd911V84555xMmjQpw4cPz0EHHZQkufLKK3PCCSdk8eLF2bx5c2bOnJkHHnigwdmhX/va13LxxRc3zYsBAACwFzrggAO2OX3NmjXp1atXLrjggnzzm9/MX//61wwbNizl5eW5+OKLU1dXl4qKigbLtG7dOuvWrUtdXV2SpE2bNlvV62uvXrb+ef3y21t3Y+y/f7tGzc+bg/cFsLMcLwBKT1VV26ZuQgPNLvirqqrK4MGDc9111+WrX/1qNm3alFNOOSWDBw/Otddeu80zNG+//fYkW84OPeecc7aq78zZoUle8+zQ/fbbb6f3wRkUAM2b4zQAb0Rz/R5pru16peOPPz7HH3988flRRx2VCy+8MHfddVcuvvjiVFRUZP369Q2WWb9+faqqqoqhXf2ILq+st23bNoVCYata/fO2bdumoqIia9as2eayjbFixZr8/+eoQsrKtvyI730B7IjjBU2tRYvyZhdOQKlYubIumzZt3q3bqP8e2RnNLvjbvHlzWrdunS996UsZOHBgFi1alE996lOpra3d4Rmau/Ps0MYEf87cAWi+/CMXgDfC98gb89vf/jbLly/Phz70oeK0l19+Oa1bt06SdO3aNfPnz2+wzIIFC3LiiSemffv26dixY4PhQJctW5ZVq1alpqYmmzdvzqpVq7J8+fJUV1cnSRYuXJgDDzww7dq1S01NTaZNm7bVurt27dqofSgU4gdbtuJ9AewsxwuA0tScju3NLvi77777cs899+Tuu+9OsqXjN2zYsIwePTpnnnnma56huTvPDm2M5nrmjrM6APbMGTilyvcIQPP+HmnMGaBNpVAoZOzYsTnkkEPSp0+f/PnPf84tt9ySK664IkkycODADBs2LB/84AfTs2fPTJw4MStWrEi/fv2SJAMGDMj48ePTvXv3VFVVZcyYMenVq1cOPvjgJEnPnj0zZsyYXHPNNVm5cmVuuOGGDBw4MEnSr1+/fO1rX8uECRMyaNCgPPLII5kyZUpuuOGGpnkxAAAAdoNmF/z985//zMsvv9xgWsuWLdOqVasdnqG5O88ObQxn7gA0b47RALwRvkdev379+uWKK67I1VdfnaVLl6a6ujqf/vSnc/bZZyfZch/3q666qljv0qVLbrrpplRWViZJhg0blo0bN2bQoEGpq6tL7969M27cuOL6a2trc8011+SUU05JeXl5+vfvn6FDhybZcluJm2++OaNHj05tbW06dOiQkSNHpk+fPnv6ZQAAANhtml3wd8IJJ+Qb3/hGvvvd7+aSSy7JkiVLMn78+Jx55pk7PENzd54dCgAAQOPV31O93oc+9KEGQ32+2tlnn10MAl+tVatWGT58eIYPH77NenV1dWpra7e77u7du+e2227biVYDAADsnZpd8NelS5fceOONGTduXL7//e+nXbt2OeusszJs2LDss88+r3mG5u48OxQAAAAAAACas2YX/CXJcccdl+OOO26btR2dobk7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d++MHj06Gzdu3DM7DQAAAAAAAG9Aswz+Pv3pT2fdunW57777cv/996dFixb50pe+lNWrV+fSSy9N//79M2vWrIwePTpjx47NnDlzkiQzZszIqFGjcu2112bWrFk566yzMmTIkLz44otJkvHjx2fatGm54447MnXq1LRu3TojR44sbveyyy5LmzZtMnXq1Nx+++156KGHMmHChKZ4CQAAAAAAAKBRml3w97//+7957LHHcu2112a//fbLvvvum1GjRmX48OG59957U1lZmUGDBqVly5bp27dvzjzzzEycODFJMmnSpJx++unp2bNnWrVqlcGDB6eqqip33XVXsX7JJZfkoIMOyr777psrr7wyDzzwQBYvXpxFixZl5syZGTFiRCoqKtKpU6cMHTq0uG4AAAAAAABozlo2dQNebc6cOenSpUt+9rOf5ac//WlefPHF/Nu//Vs+//nPZ/78+ampqWkwf5cuXXL77bcnSRYsWJBzzjlnq/rcuXOzZs2aPPvssw2Wr66uTvv27TNv3rwkSWVlZTp27Fisd+7cOUuWLMkLL7yQ/fbbb6f3oays0bsNwB7kOA3AG9Fcv0eaa7sAAADYc5pd8Ld69erMmzcvRx55ZCZPnpz169fnc5/7XD7/+c+nuro6FRUVDeZv3bp11q1blySpq6vbbr2uri5J0qZNm63q9bVXL1v/fN26dY0K/vbfv91OzwvAnlVV1bapmwDAXsz3CAAAAM1Zswv+9tlnnyTJlVdembe85S3Zd999c9lll+W8887LgAEDsn79+gbzr1+/Pm3bbul8V1RUbLNeVVVVDPHq7/f36uULhcJWtfrn9evfWStWrEmh0KhF9ogWLcr9UAG86a1cWZdNmzY3dTP2Sr5HAJr390hZmZMQAQAA3uya3T3+unTpks2bN2fDhg3FaZs3b+lYv/Od78z8+fMbzL9gwYJ07do1SdK1a9ft1tu3b5+OHTtmwYIFxdqyZcuyatWq1NTUpGvXrlm1alWWL19erC9cuDAHHnhg2rVrXOe5UGieDwC2aOrj8d76AGCLpj4eO1YDAACwPc0u+DvuuOPSqVOnfPGLX0xdXV2ef/75fOtb38r73ve+nHHGGVm+fHkmTJiQDRs2ZPr06ZkyZUrxvn4DBw7MlClTMn369GzYsCETJkzIihUr0q9fvyTJgAEDMn78+CxevDhr167N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zuzZs4uPr33ta1m9enUuvfTS9O/fP7Nmzcro0aMzduzYzJkzJ0kyY8aMjBo1Ktdee21mzZqVs846K0OGDMmLL76YJBk/fnymTZuWO+64I1OnTk3r1q0zcuTI4nYvu+yytGnTJlOnTs3tt9+ehx56KBMmTGiKlwAAAGC3EfwBAACwxzz++OM58sgjt5p+7733prKyMoMGDUrLli3Tt2/fnHnmmZk4cWKSZNKkSTn99NPTs2fPtGrVKoMHD05VVVXuuuuuYv2SSy7JQQcdlH333TdXXnllHnjggSxevDiLFi3KzJkzM2LEiFRUVKRTp04ZOnRocd0AAAClQvAHAADAHrF58+Y88cQT+cMf/pD3vve9OfHEE/OlL30pq1evzvz581NTU9Ng/i5dumTu3LlJkgULFmy3vmbNmjz77LMN6tXV1Wnfvn3mzZuX+fPnp7KyMh07dizWO3funCVLluSFF17YjXsMAACwZ7Vs6gYAAADw5vD888/n8MMPz6mnnpra2tqsXLkyn//85zNixIgccMABqaioaDB/69ats27duiRJXV3ddut1dXVJkjZt2mxVr6+9etn65+vWrct+++230/tQVrbTs/ImUP9+8L4AdsTxAqC07e7je2PWL/gDAABgj6iurm4wvGZFRUVGjBiR8847LwMGDMj69esbzL9+/fq0bdu2OO+26lVVVcUQr/5+f69evlAobFWrf16//p21//7tGjU/bw7eF8DOcrwAKD1VVY3rU+xugj8AAAD2iLlz5+bOO+/MZz/72ZT9/6esvvzyyykvL89RRx2VH/3oRw3mX7BgQbp27Zok6dq1a+bPn79V/cQTT0z79u3TsWPHBsOBLlu2LKtWrUpNTU02b96cVatWZfny5amurk6SLFy4MAceeGDatWvcD7ArVqxJofC6dp8SVFa25Ud87wtgRxwvaGotWpQ3u3ACSsXKlXXZtGnzbt1G/ffIzhD8AQAAsEdUVlZm4sSJad++fS666KI899xz+drXvpZ///d/z6mnnppvfOMbmTBhQgYNGpRHHnkkU6ZMyQ033JAkGThwYIYNG5YPfvCD6dmzZyZOnJgVK1akX79+SZIBAwZk/Pjx6d69e6qqqjJmzJj06tUrBx98cJKkZ8+eGTNmTK655pqsXLkyN9xwQwYOHNjofSgU4gdbtuJ9AewsxwuA0tScju2CPwAA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AKBKrFmzRs8884xWr16tkpISc5cDWJSDBw/q8ccfV5MmTbRx40YlJydr+fLl+te//qVhw4aptLT0lpdx8eJFGY3GKqhWys3NveU2vLy8lJycXKVt3g5r167ViBEj9P3338vHx6dCzQCqzpdffqnQ0FA98MAD2rRpkw4ePKj3339fqampCgkJUVZWloqKihjth9uC4A/AdQ0aNEj79+/XunXrFBQUpJSUFIWHh8vX11cBAQFauHChiouLJV26KjU0NFTz58+Xj4+POnTooI8//ljr169XQECAPD09NX36dFPbx48fV1hYmLp166a2bdvqscceu+6IwuzsbI0fP15+fn7y9/fX9OnTlZ+ff0d+BwCAqnd5ZPfGjRsVEBAgd3d3TZ48WUlJSerXr588PDw0ZMgQnTt3Tvn5+Zo6dap69Oghd3d3de7cWe+///412y0uLtaiRYvUvXt3+fj46MUXX9Tp06fv8NoBAKqDb7/9Vjk5OZo0aZLKysq0ffv2Cu8fOnRIAwYMkI+Pj4YPH266neDlfdi8efPk7e2tyMhISf9/wWT79u3Vt29fbd682dRWVlaWRo8ercDAQLVr107du3dXbGys6f3jx49r8ODB8vDwUN++fXX48OHb/wsAbrPp06crJCREo0aNkqOjoyTpgQce0MKFC9WgQQPTbTXT09M1evRodezYUX5+fho3bpzOnDkj6dIoucDAQC1dulSdO3eWj4+PRo4cqfz8fKWmpurFF1+UJHl4eCg5Ofmmx43nzp3T+PHj5e3tLV9fX40ZM0YXLlzQ66+/rqSkJC1btkzh4eEV1qOsrEydOnXSzp07TdMCAwM1evRo0+v58+drwoQJFUb1DRs2TBkZGZoxY4ZmzZplmnfFihUKDg6Wu7u7Ro0add1zGyUlJVq0aJG6du2q9u3b69lnn9VPP/0kSSoqKlJ0dLS6du0qb29vDR48WP/+979Nn3Vzc9PHH3+snj17ysPDQ08//bSOHDkiSfLz81NKSorefvttPffccxVqvtb2bfHixYqIiNDEiRPVvn17denSRdu2bdN7772nTp06ycfHR0uWLDEt+2bnb3bt2qXevXvL3d1dYWFhd204Ctyq4uJiTZ06VWFhYRozZoxcXFxkZWWlli1b6p133lHDhg01e/Zs06jlPn36KD4+XpJUUFCgqVOnyt/fX76+vlq4cGGFdm/U53Vzc1NUVJR8fX2v2p7h3kLwB+C6YmNj5eXlpbCwMG3evFlDhw5V69attWfPHq1du1b79u3T4sWLTfMfOHBALi4u2r9/v0aNGqW5c+cqMTFR8fHxiomJUWxsrL7//ntJ0siRI9WmTRt98cUXSkpKkr+/v2bOnHlVDWVlZRoxYoSsra21fft2ff755zpz5kyFEBEAUD19/fXXio+P1/r167Vp0ya98cYbWr58uXbt2qVffvlFa9eu1YIFC5SWlqbY2FglJydr6tSpWrhw4TUDvYULF+qrr75STEyMEhIS1K5dOw0bNky//fabGdYOAHA3+/jjj/Xkk0/K3t5eoaGhWrlyZYX3d+7cqblz5yohIUFNmzZVWFhYhVGBBQUF+uabbzRmzBjFxcVp3rx5mjp1qr7//ntNmTJFkZGR+uKLLyRJU6dOlcFg0NatW3Xw4EH99a9/1RtvvKGCggIZjUaFhYWpdevW2r9/v95+++0KAQNQHaWkpOjnn3++5m04nZyctGTJErm6uspoNGrYsGGysbHRjh07tG3bNklSeHi46fuWnp6urKwsffHFF/rss8+UnJystWvXqlmzZlq+fLkkKTk5WR4eHjc9boyIiFB+fr527NihXbt2KS8vT5GRkZo9e7bp3MfvLzCztrZWYGCg9uzZI0k6ceKEcnJy9O2336q8vFyStHv3bvXo0aPC51auXKnGjRsrMjKywvmL9PR0bdmyRdu3b9cPP/ygNWvWXPN3uHTpUm3ZskUrVqwwjcwLCwtTaWmpZs6cqb1792r16tX65ptvFBQUpKFDhyojI8P0+a1bt+qTTz7Rnj175ODgoOjoaEnSN998Y6pr9erV11z2lds3Sdq+fbsCAgJ04MAB9evXT+PGjVN+fr6+/vprzZkzR4sWLVJ6evpNz9+cOHFCERERCgsLU1JSkp544gklJCRcswaguktOTlZ2drZCQkKues/a2lqDBg3S7t27tWnTJknSli1b9Nhjj0m6dBtjb29vJSQkaNGiRVq2bJlpZG5l+rwpKSn66quvTN973JsI/gBUyldffaXi4mKNHTtWdnZ2atSokSIiIiocpNasWVNDhgyRtbW1/P39VVpaquHDh8vBwUGPPvqo7r//fqWnp0uSli1bppEjR6q8vFzp6emqW7eusrKyrlrujz/+qEOHDmnGjBmqXbu26tevr4kTJ2rr1q1cGQYA1dywYcPk4OCgNm3ayNnZWY8//rhcXFzk6Ogod3d3paena+TIkfrb3/6m2rVrKzMzU3Z2dpJkuhL8svLycq1bt05jx45Vs2bNZGdnp1deeUVGo1FfffWVGdYOAHC3Sk9PV0JCgp599llJ0pNPPqljx47pu+++M80zbNgwubm5yc7OTpMmTVJaWlqFETUhISGytbVV3bp1tWHDBj311FPq2LGjbGxs1LFjRz311FNat26dJCkqKkozZsyQwWBQRkaGatWqpaKiIl24cEHJycn65ZdfNGHCBNnZ2al169Z6/vnn7+wvBKhil59V5eTkdMP5kpKSlJqaqsjISNWpU0d169ZVZGSkfvrpJ/3444+m+V555RXZ29urRYsW8vX11cmTJ6/Z3o2OG9PT0/Xdd99p4sSJql+/vmrXrq158+bp5Zdfvun6BAUFmYK/vXv36rHHHlNZWZkOHz6s48eP68yZM/L396/U72bkyJGys7OTi4uLvL29lZKScs35/vGPf+iFF15Qq1atZGNjo5dfflmLFi1SUVGRtmzZonHjxqlFixaytbXVkCFD9OCDD2rLli2mzw8ePFjOzs6qU6eOevXqZRq1XBlXbt8kqVWrVvrLX/4iKysr+fn5qbS0VOHh4TIYDKbnkWZkZNz0/E18fLweeeQR9evXTzVq1FBQUBCPlIHFutxfvd528P7775fRaLzmuc3WrVurf//+srKyUocOHeTk5KSUlJRK93n79OkjBwcH03cY96Ya5i4AQPWQnp6uc+fOydvb2zStvLxcRqNROTk5kqR69erJyspK0qWrVyRV2MlYW1urrKxMkvTTTz9pxIgROnv2rFq2bClHR0fT1XJXSktLU2lpqbp27Vphuq2trVJTU1W/fv2qXVEAwB1Tr1490882NjZX7TPKy8uVk5Oj2bNn6/Dhw2ratKkeeeQRSTLtTy47d+6cfv31V0VERJj2QZJkNBpNF50AACBder5VSUmJ+vfvb5pWUlKilStXysfHR5LUtGlT03sODg6qV6+esrKydP/990uS6X/p0q3tmjVrVmEZTZs21e7duyVJqampio6O1qlTp+Tq6qoWLVpIurQvy8rKUv369WVvb2/6bPPmzat4jYE7y9nZWZJ09uxZubq6XvV+dna2nJyclJOTYwrhLqtdu7bq1aun9PR00wnzy+1JksFg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",
      "text/plain": [
       "<Figure size 1800x600 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "cols = ['Sex',\t'Known_contact']\n",
    "\n",
    "# Set up the figure and axes\n",
    "\n",
    "fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))\n",
    "\n",
    "# Plot univariate distributions for each column\n",
    "for i, col in enumerate(cols):\n",
    "    sns.countplot(data=df , x=col, ax=axes[i])\n",
    "    axes[i].set_title(f'Distribution of {col}' , fontsize=18 )\n",
    "    axes[i].set_ylabel('Count')\n",
    "    axes[i].tick_params(axis='x')\n",
    "    axes[i].grid(True)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 121,
   "id": "db82a81c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABv4AAAJOCAYAAAB/dnBOAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAAC01UlEQVR4nOzdeVxUZf//8fewKIgEKIqalrkMlkshBm6pWWS5515karkklLfet5aWpWlu3WVGpZVlZFrduZWami2WSy5kpn3NDS0ld1CQRWQ7vz/8zckRUEDW4fV8PHg8ZK5znXPNOXOGt/OZcx2LYRiGAAAAAAAAAAAAAJRpTiU9AAAAAAAAAAAAAAA3jsIfAAAAAAAAAAAA4AAo/AEAAAAAAAAAAAAOgMIfAAAAAAAAAAAA4AAo/AEAAAAAAAAAAAAOgMIfAAAAAAAAAAAA4AAo/AEAAAAAAAAAAAAOgMIfAAAAAAAAAAAA4AAo/AEAAAAAAAAAAAAOwKWkB4Dy5++//9Z9992XY5uzs7MqVqyoatWq6Y477lD37t3VsWPHHJfdvn27Hn/8cUnS3r175eJyYy9nwzB05MgR1a9fP1/9Bg4cqB07duipp57SmDFjJNk/x/Xr1+vWW2+9obHl1cWLFxUXF6fatWubjy1fvlwTJkyQn5+fNm7cWCzjKE4//fST5s6dq4MHD0qSbr75Zq1atUoWiyVP/TMzM/Xdd99p7dq12rNnj86ePSuLxaLq1asrICBAffr0UXBwcFE+hSL11ltv6e2331bz5s312WeflfRwHNbhw4dVr169PL/uAACFi3xZdMiXecuXtmOWV++8847uv//+Gx4rAKBsI8MUHTJM/j4ji4uL02effaaNGzfq2LFjSkpKkpeXl2rXrq02bdqof//+8vPzK+qnUOIyMjIUExOj2267zXyssM+vklbQ8xtlS9l+laLMs1qtqly5svl7VlaWEhMTFRMTo6NHj2rt2rVq27at5syZI09PzyIbx549ezR16lTdeuuteu2114psO0Vp1apV+u9//6tnnnlGffv2LenhFIs9e/boqaeeUlZWlry9vVW7dm35+fnlufhy5MgRjRkzRvv375ckVapUSXXr1lVGRob+/vtvrVy5UitXrtSDDz6oGTNmqFKlSkX5dFAGJSUlafbs2frf//6n3bt3l/nwBwCOgHxZeMiX+c+XVatWzdMHmt7e3jc4UgCAoyHDFB4yTP4yzA8//KBnn31WiYmJcnV1Vd26dXXrrbfqwoUL+v333/Xbb79pwYIFeumll9SrV69ieDYlY/PmzXrllVfUqVMns3DtaBzh/Ebe8AklStTEiRNzvJoqLS1NX331lWbMmKHNmzcrLCxMH374oSpUqGAu06xZM61Zs0aSbvjD9k8//VR79uwp0LeOZs2apYsXL8rHx+eGxnCj3njjDZ0+fTrb4yEhIbrzzjvl6upaAqMqWt98842ysrJUu3Ztff3113Jzc8tz36ioKA0fPlwpKSlq0qSJRo0apfbt25vtqamp+vzzz/Xmm29q3bp1On/+vBYsWEBhB3b27t2rxYsXl/QwAABXIF8WHvJl/vKlJLVr104zZ84sotEBABwZGabwkGHynmEOHz6s0aNH69KlS3r66af1xBNPyMPDw2w/d+6cXn/9dS1dulQvvPCCatSoodatWxfV0yhR7733nv7888+SHkaRupHzG2UL9/hDqVShQgX17dtX7733npydnbVjxw59/PHHdsu4u7urfv36JX5Zcq1atVS/fn1VqVKlRMeRG09PT9WvX1+33HJLSQ+l0J0/f16SdOedd+brQ5nz58/rP//5j1JSUtS2bVt9+umndkU/SXJzc9PgwYP1zjvvyGKxaPv27Vq0aFGhjh8AABQf8mXhIV8CAFB8yDCFhwyT3YIFC3Tp0iV17txZzzzzjF3RT5KqVKmiqVOnqlWrVsrKytLbb79dqOMGUDQo/KFUu/vuu9W/f39J0gcffKDU1NQSHhFKk6ysLEmy+5ZbXsyZM0enT59WpUqV9N///lcVK1bMddnWrVvroYcekiR99NFH5jYBAEDZRL7EtRQ0XwIAUNTIMLiWgmaY33//XdLlgmFunJyc1KdPH3N5PhsDSj8Kfyj1bKEmPj5ev/zyi/n49u3b5e/vL39/f2VkZNj12b17t8aMGaNOnTqpWbNmuvvuu9WvXz+9//77SkpKyraOFStWSLo8B7i/v78GDhwo6fINiP39/dWmTRudPn1aw4cPV7NmzRQUFKT//Oc/ki7fuNjf319vvPFGjuPPysrSJ598oq5du6pp06Zq3bq1/vWvf5l/WK90redkY2vfvn27JOmtt96Sv7+/jh8/Luny1BD+/v566623JF2+cbG/v7/atWuX4/q2bt2qZ555Rm3btlWTJk3UsmVLDR06VOvXr89x+Y4dO8rf31+HDx/Wjh07NHz4cAUHB6tp06Z66KGHFBERoeTk5Bz7XktmZqaWLFmigQMH6u6771aTJk3UoUMHjRs3Tnv37rVb1vacbcdtxYoV5n75+++/r7mdS5cuaeXKlZKkPn365OlbaOHh4YqIiNCyZcvk5GT/tpmfcUv/vKb8/f119OjRHLdn28fLly/P1nbkyBE9//zzuu+++9S0aVN17NhRs2bNUmJiovlatL02rnbu3DnNnDlTISEhatKkidq0aaNRo0bl+Fq8EYcPH9aECRPUpUsX3XXXXQoMDFTPnj31xhtvKC4uzlzuwoULatasmfz9/XN9vUnS4MGD5e/vr/fff1/SP8f/jTfe0NmzZzVp0iS1a9dOTZs21f3336833nhDaWlpki6fU08++aTuvvtuNWvWTA8//LC+/PLLbNsYP368/P39FRkZqcOHD+vpp59WcHCwAgIC1LNnT3388ce6dOmSXZ+OHTuaN3eWpMaNG+f4GizoOXb06FFt3rxZgwcPVosWLdS8eXMNGDBAP/74o6TLN5z+4IMP1LVrVzVr1kzBwcF6+umndfjw4WzrzMzM1KeffqqBAwea42jbtq3Cw8O1YcOGXPc9ADgq8qU98uWN5cvC8t1332n48OFq1aqVmjRponvuuUf/+c9/so01JiZGjRo1kr+/v/74449c19epUyf5+/tryZIl2fpPnjxZISEhatq0qVq0aKFHH31US5YsUWZmZrb12HLSZ599pv/973/q0KGDmjZtqgceeCDX3AkAKBpkGHtkmBvPMLZpT3/88UcZhpHrch07dtRXX32lb7/9NttnY5J0+vRpzZo1S507d9add96pgIAA9ejRQ2+//bYuXLiQbXnbuF977TV999136tSpk5o0aaKOHTvq66+/NpdLS0vTxx9/rP79+yswMFDNmjVTp06dNGPGDJ05c+a6zy8vbK+LHTt2SJLeffdd+fv7a/z48dmWzcjI0Mcff6yHH35Yd955p+6++2499thj5lS7V7K9hvv166fDhw/rkUceUdOmTdWqVSvNmjXLbtlvvvlGQ4cOVcuWLc3PbJ555hlt3bo113GfOXNGb7zxhvr06aOgoCA1btxYQUFBGjBggBYsWGD35YDrnd9wPBT+UOo1atTIvLmx7Q34WtavX69HH31Ua9asUUJCgho0aKAqVapoz549ev311zVgwAAz2Hh6eqp58+aqWrWqpMuXrzdv3lxWq9VunWlpaXryySf1888/q379+rJYLLr55pvzNP4XX3xRr7zyis6ePSur1aq0tDStW7dO/fr107Jly/KzK3JUs2ZNNW/e3PxGz6233qrmzZurZs2a1+07depUDR48WOvXr1d6eroaNWokV1dXbdq0Sc8884xGjx6t9PT0HPsuWbJEjz/+uLZt2yY/Pz9VqVJFR44c0TvvvKMnn3wyxw8NcpOUlKTQ0FBNnDhRO3bskKenp/z9/ZWYmKiVK1eqT58++uijj7I9Z9txq1q1qpo3b67mzZtf8+o9Sdq1a5dSUlIkSW3atMnT+Bo0aKBOnTrJ19f3hsZ9ozZt2qTevXtr2bJlOnfunPl6WrBggfr27WtO65CTM2fO6OGHHzbHc9tttykhIUHffPONHnnkkWsGifzYtWuX+vTpo+XLl+vMmTO67bbb5Ofnp4MHD+rdd9/Vww8/rJMnT0qSbrrpJoWEhEiSvvrqqxzXd+rUKW3fvl3Ozs7q0aOHXdvRo0fVo0cPffHFF/Ly8pKvr69iYmL07rvvavz48frss880aNAg/frrr6pdu7YqVqyoP/74Q88995w+//zzHLd34MAB9e3bV999952qV6+uGjVqaN++fZo+fbqGDBmixMREc9kmTZrYvVfk9Bq8kXPs448/1pNPPqm9e/eqTp065v596qmn9O233+qJJ57Qf//7X124cEG33XabLly4oG+//VaPPvqo3b0MDMPQmDFj9PLLL2vHjh2qVKmS/P39lZmZqe+++05PPfWU3nzzzVyPKQA4IvLltZEv85cvb1RGRobGjh2r8PBw/fTTT7JYLPL391daWppWr16tvn372k05X6dOHQUFBUmS+YW2q/3222/666+/5O7ubs5eIUnffvutunbtqs8++0xnzpxRvXr1VKVKFe3cuVMTJ07Uk08+mesHlCtXrtRLL70kwzBUt25dnT17Vrfffnsh7gkAwPWQYa6NDJP/DNO2bVtJl4ueAwcO1DfffGN+bnalSpUqqVGjRqpRo0a2tq1bt6pLly5asGCBjh07pttuu00333yzDh48qLfeekvdu3fXgQMHctx+VFSURo0apQsXLqh+/fo6c+aMmS/OnDmjfv36afr06dq9e7e8vLzUoEEDnTx5UpGRkerWrZt27tx5/Z16HbZ9Zju3bPu0bt262ZYdOnSopk+fbuaozMxMRUVFacyYMblOg3ru3DkNGjRI+/btU4MGDXTp0iVz3enp6Xr66ac1atQobdq0SS4uLmrUqJEyMjK0fv16DR48WNOnT8+2zt9++01dunTRu+++q4MHD6p69epq2LChsrKytGvXLs2aNcvutZfX8xsOxACKWUxMjGG1Wg2r1Wps27YtT326detmWK1W4z//+Y/52LZt28z1pKenG4ZhGJmZmUabNm0Mq9VqzJ8/38jIyDCX/7//+z+jZcuWhtVqNd577z279T/33HPZ1n/1WIOCgoxDhw4ZhmEYly5dMhITEw3DMIzHHnvMsFqtxuzZs3PsZ7VajTfeeMNIS0szDMMwUlNTjcmTJxtWq9Vo3LixER0dfc3ndLXc9t29995rWK1W44svvrB7fNmyZYbVajXuueceu8c//PBDw2q1GnfccYexaNEiIzMz02xbs2aNcddddxlWq9WYOnVqjtuxWq3G+PHjjQsXLhiGYRhZWVnGokWLzLZvv/02x/HnZMSIEYbVajXatGlj97wuXbpkzJkzx1znN998Y9fPdtyee+65PG9r8eLF5vpOnjyZ536FNe4rXxt//fVXjuu17eNly5aZj8XGxhotWrQwn29SUpJhGJf3+9KlS43GjRvn+NqIiIgwH7/33nuN3bt3m23Hjh0zHnjgAcNqtRp9+vS5oX1h07dvX/N1c+nSpRy39eKLL5qP//zzz+a5cP78+Wzre/fddw2r1WoMGzYsx+f00EMPGX/++afZ9uabb5ptjRo1MmbMmGGOIyUlxXjyySfNfXEl22vJ1vbHH3+Ybb/++qvRunXrbGM3jGufs4Vxjk2dOtW4ePGiYRiGkZCQYL4XNmrUyAgKCjI2btxo9tu3b58REBBgWK1W46233jIf/+mnnwyr1Wq0bNnS2L9/v/l4RkaGuX/vuOOOGz4fAKAkkS/Jl1crznxpGP8cs/z2MwzDeO211wyr1Wq0a9fO7m97RkaGsXDhQuOOO+4w/P39jc2bN5ttK1asMKxWq9G2bVu7/Wxjez2MGzfOfGzfvn1GkyZNDH9/f+ONN94wUlNTzba9e/eaWW3s2LF267oyJ02ZMsU8B+Li4vL9XAEA9sgwZJirFXeGuXDhgtG5c2e7Y9S4cWNjwIABxmuvvWb89NNPRkpKSq79//77b3P/PPXUU8bZs2fNtmPHjhn9+/c3rFar0aFDB3P/GIb9Zzvh4eHmZze2fJGVlWX2feSRR4zDhw/bjXnChAmG1Wo1goODjTNnzuTrOecmp9evYdi/Fps1a2asWLHCfI0kJSUZYWFh5n6znQtX97v//vuNU6dOmX1sOWzKlCmG1Wo17rrrLmPt2rVm34yMDGPRokXGHXfcYVitVuOjjz6ya7v//vsNq9VqhIWFGfHx8WZbWlqa8d5775nb3bBhg91zye38huPhij+UCbYby8bHx19zuXPnzuns2bOSpH79+snZ2dlsa9y4scaMGaP7779f3t7e+R7Do48+qgYNGki6PF+27Vsg19O1a1eNHj3avHS+YsWKeumllxQYGKj09HQtWLAg32O5UZcuXdK8efMkSaNGjVJoaKjdZfoPPfSQXnnlFUnSp59+muPUAI0aNdL06dPl6ekpSbJYLAoNDZW/v78k5fkbN7/99ps51WBERISCg4PNtgoVKuhf//qXOZXFa6+9lt+nmk1CQoL57xu52XRxj3vBggXm1JjTp083zwmLxaLevXvrX//613XX8dprr6lZs2bm73Xq1NHTTz8t6fIc7VdPZVkQ+/fvlyT17t3bbl75OnXq6LnnntO9995r903Ali1b6uabb1Z6enqO0yLYpuXs1atXjtubMWOG3Tewhg0bZp73gYGBGj9+vDkOd3d3DR8+XJJ0/Phxu9eCjZOTk+bOnWv37fWAgABzCoYlS5bYXU2Xm8I4xxo0aKDnn3/evCn3TTfdpNDQUEmXp0d59tlndc8995jLN2rUSJ06dZIku6k/bMckICDAPD8lydnZWSNGjNCDDz6oLl265Lg/AMCRkS8LV3nOl1e6cnqt3H6uFBcXp8jISEnS3Llz7f62Ozs7a+DAgRo8eLAMw9CcOXPMtk6dOqly5co6c+aMtm3bZrfOK3PVlRnqrbfeUlpamh577DGNHj3a7iqAO+64QxEREXJ2dtaqVasUHR2d7blVrFhR//nPf8xz4EayNACg4Mgwhau8ZxhPT099+umn6tWrl/kaSU9P16+//qr3339fw4YNU8uWLTVu3DidOHEiW//33ntPKSkpslqtevPNN+1my6pTp47ee+89VatWTSdOnNAnn3yS4xiee+4587MbW774/vvvtWvXLlWvXl0ffPCB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4GzevFlTpkzJccxPPPGEPDw8tGvXLk2ePNnuQ59vv/3WfG2XpEaNGslqtSozM1P//ve/derUKbMtLS1Nb7zxhpKSklSpUiW1aNEiW3/bN9M//vhjxcfH695771WVKlWKbfzS5alCwsLC7OaP3759uyZMmCDp8n+urpx7/cpps658/yhN51iPHj0kSf/73//05Zdf2r3ODx06pE8++USSzNczADgq8iX5Uiq6fFlQN998s/r27StJ+ve//60ffvjBbMvKytKSJUv09ttvS5JCQ0NzvHeNbYaEN998U4Zh6MEHH8zxXsHPPPOMnJ2dtXr1as2YMcPuqr6DBw9q+PDhSk9P180331zsX74CAOSODEOGkYomw3h7e5tFs+XLl2vo0KHas2eP3ecGWVlZ2rp1qx5//HElJiaqc+fOCggIMNuHDRsmDw8PHTx4UP/617/sinQxMTEaMWKEYmNj5efnp0GDBuV5bJ07d5bVatWFCxf05JNP2l35l5SUpMmTJ+vnn3+WxWIxj9+Nys/rpjDUrFlT/fr1k3R5StMrv3iVlZWlTz/9VG+99ZYkKSwszPw8yvYa3LJli3755RezT0ZGhlavXq0xY8aYj6WmptptsyjPb5Qu3OMPJeqVV16x+89rRkaGLly4YDeNTceOHfXf//7XborE3FSoUEFvvvmmnnzySa1Zs0bff/+9brnlFjk5OSkmJkYpKSlyd3fXzJkz7eYUv/322yVdvt/Fgw8+qAYNGpj/wb4Rrq6uatu2raZMmaJ58+bJz89PR44cUUpKijw8PPTmm2/aXYbt5OSk0aNH68UXX9SOHTvUvn173XbbbTpz5ozOnj2re++9V+fOnTOnabjSHXfcoYMHD+qDDz7Qxo0bFRISorCwsFzH9uSTT+rvv//WZ599pilTpuitt95SnTp1dOrUKfMS+U6dOumFF1644f1wPa+++qqeeuop7dq1S48//rhuvvlmValSRX/++aeSkpLk7Oys0aNHq0uXLoW2zXvvvVcffvihRo0apd9//11Dhw7VTTfdpNq1aysjI0N//vmnOW9+165dNW3atGzfpivouEePHq2RI0cqOjpa999/vxo0aKCEhAQdP35cTZs2VfPmzfX999/b9alZs6ZeffVVjR49Wp9//rlWrlyp+vXrKy4uTidOnNDtt9+u6OhopaenX/dbf0XpjTfe0IABA7Rjxw7df//9ql27ttzd3fX333/rwoULcnZ21pQpU3Is6DVu3FiNGjUyp9Qs7mk+pcvTTuzbt0/333+/rFarUlJSzCkVunbtqhEjRmRbvlKlSkpJSVG/fv1Uu3ZtTZs2TY0aNSo159gDDzygfv366YsvvtBzzz2nWbNmqWbNmkpKStKxY8dkGIaaNWump556qkjHAQDFhXxJvpRKJl8W1PPPP6/Tp09rw4YNGjlypKpXry4/Pz8dP35c586dk3R5v40ePTrH/vfee6+8vb3ND3Rzy1CBgYGaOnWqJk2apMjISH3++eeqX7++kpOTdfToURmGIV9fX3344Yc53n8JAFC0yDBkGKn4M8xjjz2mjIwMzZkzR5s2bdKmTZvk4+Nj3nLm+PHj5j36HnzwQc2cOdOuf506dRQREaF//etf+uGHH9S+fXs1aNBAmZmZio6OVlZWlmrVqqW33347X1/udnV11dy5czV06FDt27dPXbt21W233SZ3d3f99ddf5nSnEyZMyNeVhNdyxx13aMOGDVq1apUOHDigFi1aaNKkSYWy7txMmDBBp0+f1vfff69//etfql69umrUqKGYmBhz+tXHHntMw4YNM/s88cQTWr16tc6fP6/Q0FDVrVtXHh4e+vvvv5WQkKBKlSopICBAu3btsvtSvlR05zdKHwp/KFEHDx60+93JyUnu7u6qV6+emjZtqu7du6tVq1b5WmdwcLCWLFmijz76SDt37tRff/0lFxcX1ahRQ23bttUTTzyhWrVq2fXp2bOnjhw5oq+++kp///230tPTc51bOT8sFoveeustzZ8/XytWrNCBAwfk4+Ojzp0766mnnlKdOnWy9enXr59q1KihyMhI7dmzR4cPH9Ztt92mESNGKDQ0VKGhoTlu67nnntPFixf1888/68iRIzp8+PB1xzZ58mTdf//9+uyzz/Tbb79p37598vHx0b333qs+ffro/vvvv+F9kBfe3t765JNPtGLFCq1cuVIHDhzQ2bNn5efnpwcffFChoaHmVXKFKTg4WOvXr9fSpUv1008/6fDhwzp06JCcnZ1Vq1YtBQYGqnfv3jlenXYj427Xrp0+/fRTvf/++/rll18UHR2t2rVra9SoURo2bJief/75HLd3//33a9myZXr33XcVFRWl/fv3y8/PTyNGjNCIESMUHBwsSTle5l9cGjRooBUr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6tbtYXXu3E1//fWn5s59U9HRhxQfH69atWpp5MhRatPmHo0ZE67Tp0/ptddmaP/+P9Shw30aNeopbd78i6ZOfUmZmZmaPHmaOc6XXpogLy9v/ec/z+n48b/15puva+/ePXJzc9cDDzykJ54YLldX15LajQAAoAiRpchSAACg4MhSZT9LFd81mwAAwGFs2bJZH330qT77bLnOnz+njz/+UIcOHdT48f/WY48N1tdff6/nnntBERGva/v2rZKkTz/9ROvXr9Wbb87TsmWrVbmyp37/fY+5zokTn1W9eg30xRdf6ZtvflRQUEu9/vpMSdIbb7wjP78aGjt2gv797+fsxtK9+8PatOlHJScnSZISExO1efNGde3aQxcvXtS//jVS9erV1/LlazR37gf65Zcd+vDD94plPwEAAOSELAUAAFBwZKlro/AHAADyLTR0kDw9PVWlSlW1bn2PYmKO6auvlqtt2/Zq376jnJ2d1bTpnerW7WEtW/aFJOnrr79SaOgg3XZbPVWoUEHDho1U1apVzXW++uocPfHEcGVlZenkyRPy9LxJZ8+eue5Y7rwzQH5+NbRhw3eSpO+++0a33nqr/P0b6eefNys9PV0jRoSrYsWK8vOroWHDRmr58iVFs2MAAADygCwFAABQcGSpa2OqTwAAkG9XBiMXFxdlZWXp1KkT+vXXX/Tggx3MtszMLN18c21J0pkzp1WzZi2zzdnZWX5+Nc3fbd/MOncuTrfeepu8vb1lGHmb471r155at26NunbtqTVrVqlr156SpFOnTig+/rweeuhec1nDMJSRka7z58/Jx6dKQZ4+AADADSFLAQAAFBxZ6too/AEAgEJRrZqfHnqoq8aNe958LDY2VtLlkOTnV0OnTp002wzD0Jkzp///cmf10kvjNW3af9W2bTtJ0o8/fq+fftqQp20/9FBXffDBPEVFbdfhw9EKCXnQHNPNN9fWp58uM5dNSUnWuXPn5O3tc0PPFwAAoDCRpYrP/v37NWvWLO3du1eurq5q06aNxo8frypVqmjSpElatmyZ3X13xo8fr/79+0uSVqxYoblz5+rs2bOqV6+eXnzxRQUEBEi6fO+g1157TV999ZUuXryoli1b6uWXX1b16tUlSXFxcXrxxRe1Y8cOOTs7q3v37nruuefk4nL547ndu3frlVdeUXR0tHx8fDRy5Ej17du3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jTfeSJLMmTMnS5YsyQMPPJDFixenpKQkU6dObW738ssvT5cuXbJ48eLMnz8/jz32WKqrq9viTwAAAAAAAACt0m6Dvw996EM7rX/44YdTVlaWsWPHpmPHjhkyZEjOOuus3HvvvUmS+++/PyNHjkz//v3TqVOnjBs3LuXl5Vm0aFFzffz48Tn00EOz//7756qrrsqjjz6ampqarF69OsuXL8/kyZNTWlqaXr16ZeLEic3HBgAAAAAAgPas3QV/jY2NefbZZ/PII4/kjDPOyGmnnZarr746GzduzMqVK9O3b98W2/fu3TsrVqxIkqxateod65s2bcrLL7/col5RUZHu3bvnueeey8qVK1NWVpYePXo0148++uisWbMmr7/++h58xwAAAAAAAPCn69jWHXi71157Lcccc0xGjBiRqqqqrF+/PlOmTMnkyZNz8MEHp7S0tMX2JSUl2bJlS5Kkrq7uHet1dXVJki5duuxU31F7+747Xm/ZsiUHHHDAbr+HoqLd3pQCUlTk3MO+YMd17nrf9zjnAAAAALR37S74q6ioaDG9ZmlpaSZPnpzzzz8/lZWV2bp1a4vtt27dmq5duzZvu6t6eXl5c4i343l/b9+/qalpp9qO1zuOv7sOOqhbq7anMJSVte7fCbB381kPAAAAALQ37S74W7FiRR566KFceeWVKfrvr9Zv27YtxcXFOf744/Od73ynxfarVq1Knz59kiR9+vTJypUrd6qfdtpp6d69e3r06NFiOtC1a9dmw4YN6du3bxobG7Nhw4asW7cuFRUVSZLnn38+PXv2TLdurbu5W1u7KU1N7+ntv6sOHYpTXi5gao82bKhLQ0NjW3cD2MOKit4M/fbkZz3t045zDwAAAADtVbt7xl9ZWVnuvffe3HXXXWloaMiaNWty880355xzzsmIESOybt26VFdXp76+PkuXLs3ChQtz7rnnJknGjBmThQsXZunSpamvr091dXVqa2szfPjwJEllZWXmzJmTmpqabN68ObNmzcrAgQNz+OGH58gjj0z//v0za9asbN68OTU1NZk9e3bGjBnT6vfQ1LTnFtqvPXneLRZL+1pc8/vuAgAAAADtWbv7xV/Pnj1z55135rbbbsucOXPSuXPnjBw5MpMnT07nzp0zb968zJw5M1VVVTnwwAMzderUDB48OEkyZMiQTJs2LdOnT88rr7yS3r17Z+7cuSkrK0uSXHbZZWloaMjYsWNTV1eXQYMG5fbbb29uu6qqKtdee22GDRuW4uLijB49OhMnTmyDvwIAAAAAAAC0TrsL/pJk4MCBue+++3ZZO+64496xliSjRo3KqFGjdlnr1KlTJk2alEmTJu2yXlFRkaqqqtZ3GACgHVm0aFEmTZqUzp07N68788wzc/PNN+epp57K9ddfn1WrVqW8vDwTJkzIeeed17zdggULMnv27KxduzZHHXVUrr766vTr1y9Jsn379txyyy158MEH88Ybb2Tw4MGZMWNGDjnkkCRJbW1trr766ixfvjwdOnTI2WefnSlTpqRjx3Y55AQAAAAoOO1uqk8AAP40zzzzTEaNGpUnn3yyebn55puzcePGXHzxxRk9enQef/zxzJw5MzfccEOefvrpJMmyZcty3XXX5cYbb8zjjz+es88+OxMmTMgbb7yRJJkzZ06WLFmSBx54IIsXL05JSUmmTp3a3O7ll1+eLl26ZPHixZk/f34ee+yxVFdXt8WfAAAAAGCfJPgDACgwzzzzTD70oQ/ttP7hhx9OWVlZxo4dm44dO2bIkCE566yzcu+99yZJ7r///owcOTL9+/dPp06dMm7cuJSXl2fRokXN9fHjx+fQQw/N/vvvn6uuuiqPPvpoampqsnr16ixfvjyTJ09OaWlpevXqlYkTJzYfGwAAAIA9z7xLAAAFpLGxMc8++2xKS0tz1113Zfv27Tn99NMzadKkrFy5Mn379m2xfe/evTN//vwkyapVq3LuuefuVF+xYkU2bdqUl19+ucX+FRUV6d69e5577rkkSVlZWXr06NFcP/roo7NmzZq8/vrrOeCAA3b7PRQVtfptUwCKipx72BfsuM5d7/se5xwA4P0h+AMAKCCvvfZajjnmmIwYMSJVVVVZv359pkyZksmTJ+fggw9OaWlpi+1LSkqyZcuWJEldXd071uvq6pIkXbp02am+o/b2fXe83rJlS6uCv4MO6rbb21I4ysq6tnUXgPeRz3oAANgzBH8AAAWkoqKixfSapaWlmTx5cs4///xUVlZm69atLbbfunVrunbt2rztrurl5eXNId6O5/29ff+mpqadajte7zj+7qqt3ZSmplbtsts6dChOebmAqT3asKEuDQ2Nbd0NYA8rKnoz9NuTn/W0TzvO/d5g0aJFmTRpUjp37ty87swzz8zNN9+cp556Ktdff31WrVqV8vLyTJgwIeedd17zdgsWLMjs2bOzdu3aHHXUUbn66qvTr1+/JMn27dtzyy235MEHH8wbb7yRwYMHZ8aMGTnkkEOSJLW1tbn66quzfPnydOjQIWeffXamTJmSjh3dvgMAdp9n/AEAFJAVK1bklltuSdNb7qZu27YtxcXFOf7447Ny5coW269atSp9+vRJkvTp0+cd6927d0+PHj2yatWq5tratWuzYcOG9O3bN3369MmGDRuybt265vrzzz+fnj17plu31t3ka2racwvt15487xaLpX0trvl9d9lbPPPMMxk1alSefPLJ5uXmm2/Oxo0bc/HFF2f06NF5/PHHM3PmzNxwww15+umnkyTLli3LddddlxtvvDGPP/54zj777EyYMKH5y1Bz5szJkiVL8sADD2Tx4sUpKSnJ1KlTm9u9/PLL06VLlyxevDjz58/PY489lurq6rb4EwAAezHBHwBAASkrK8u9996bu+66Kw0NDVmzZk1uvvnmnHPOORkxYkTWrVuX6urq1NfXZ+nSpVm4cGHzc/3GjBmThQsXZunSpamvr091dXVqa2szfPjwJEllZWXmzJmTmpqabN68ObNmzcrAgQNz+OGH58gjj0z//v0za9asbN68OTU1NZk9e3bGjBnTln8OAIBWe+aZZ/KhD31op/UPP/xwysrKMnbs2HTs2DFDhgzJWWed1Tzbwv3335+RI0emf//+6dSpU8aNG5fy8vIsWrSouT5+/Pgceuih2X///XPVVVfl0UcfTU1NTVavXp3ly5dn8uTJKS0tTa9evTJx4sQWMzkAAOwOcwUAABSQnj175s4778xtt92WOXPmpHPnzhk5cmQmT56czp07Z968eZk5c2aqqqpy4IEHZurUqRk8eHCSZMiQIZk2bVqmT5+eV155Jb17987cuXNTVlaWJLnsssvS0NCQsWPHpq6uLoMGDcrtt9/e3HZVVVWuvfbaDBs2LMXFxRk9enQmTpzYBn8FAID3prGxMc8++2xKS0tz1113Zfv27Tn99NMzadKkrFy5Mn379m2xfe/evTN//vwkb86UsOMLVW+tr1ixIps2bcrLL7/cYv+Kiop07949zz33XJI3v8DVo0eP5vrRRx+dNWvW5PXXX2/V85IBgH2b4A8AoMAMHDgw99133y5rxx133DvWkmTUqFEZ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AAAAAIAC0LGtOwDtxdChQ7N27dp07NjysujXr1/mzZv3R/f94Ac/mHvuuSeDBg3ak10E9oBrrrkmCxcuTJI0NDSkvr4+paWlzfW5c+dmwIABbdU9gL2GsRTsm4ylAP48jKVg32QsxZ5Q1NTU1NTWnYD2YOjQofn85z+fysrKVu9rgAWF4fvf/36++c1v5qc//WlbdwVgr2MsBRhLAbx3xlKAsRR/Lqb6hN3wyiuv5PLLL8/QoUNzwgknZNiwYZk/f/4ut/3JT36SkSNHpn///vnYxz6W2bNnN9fWrVuXSZMm5ZRTTsmpp56aa665Jps3b36/3gbQSi+++GI++MEP5sYbb8zJJ5+cGTNm5I477siFF17YYruhQ4fm+9//fpJk27Zt+cY3vpFhw4Zl4MCBGT9+fFavXt0W3QdoN4ylYN9kLAXw52EsBfsmYyneK8Ef7IapU6emU6dO+dGPfpRf/vKX+Zu/+Ztcd911qaura7Hd1q1bM3ny5FxzzTV54okncuutt2bu3Ll5+umn09jYmIkTJ6a4uDg/+clPsnDhwrz66qu55ppr2uhdAburrq4uS5YsyRVXXPGu237961/PI488kurq6ixevDgnnHBCLrroovzhD394H3oK0D4ZS8G+zVgK4E9jLAX7NmMpWssz/uAtZsyYkVmzZrVY9+ijj+b6669P165d06lTp6xZsyZdu3bN1q1bs3HjxnTt2rXF9iUlJZk/f34aGxtz0kkn5YknnkhxcXGefvrpPPvss7n77rub95kyZUo++tGP5uqrr055efn79j6B1hk9enT222+/7Lfffn90u6amptx3332pqqpKr169kiSXXXZZvve97+WRRx7JiBEj3o/uArQZYylgV4ylAHaPsRSwK8ZStJbgD95i2rRpu5xL/Te/+U1uuumm/O53v8uRRx6ZI444IknS2NjYYruSkpL8y7/8S2bPnp0rr7wymzdvzogRIzJ16tS8+OKL2b59e04//fQW++y3336pqakxwIJ27JBDDtmt7V577bVs2bIlX/rSl1Jc/P9+VF9fX5+XXnppT3UPoN0wlgJ2xVgKYPcYSwG7YixFawn+4F3U19fnkksuyZe//OVccMEFKSoqyq9//ev88Ic/3GnbzZs359VXX82tt96aJPnP//zPfPnLX863v/3tDB8+PCUlJVm2bFk6dOiQ5M05l2tqapoHbED7VFRU1PzfxcXFqa+vb37d2NiYDRs2JEnKy8vTuXPnzJs3LyeeeGLzNv/1X/+VHj16vF/dBWhXjKUAYymA985YCjCWorU84w/eRX19fbZu3ZqSkpIUFRVlzZo1ufnmm5trb1VXV5fx48dn4cKFaWpqyiGHHJLi4uKUl5fn+OOPzxFHHJEbb7wxdXV12bp1a2bNmpVx48Zl+/btbfHWgPfg6KOPznPPPZeVK1emoaEhd911V7Zs2ZLkzcHXmDFjcuutt+bll19OY2NjFixYkE984hMepAzss4ylgLcylgJoHWMp4K2Mpdgdgj94F126dMmsWbPyrW99K/369ctnPvOZnHLKKamoqMj//b//t8W2PXr0SFVVVebOnZuTTjopn/jEJzJ48OCMGzcuHTt2zJ133pl169blr//6r3PqqafmhRdeyN13353OnTu30bsDWuvMM8/MWWedlXHjxuUjH/lI1q9fn/79+zfXp0yZkhNOOCEXXHBBBgwYkOrq6lRVVeWYY45pw14DtB1jKeCtjKUAWsdYCngrYyl2R1FTU1NTW3cCAAAAAAAA+NP4xR8AAAAAAAAUAMEfAAAAAAAAFADBHwAAAAAAABQAwR8AAAAAAAAUAMEfAAAAAAAAFADBHwAAAAAAABQAwR8AAAAAAAAUAMEfwF5i06ZNee2119q6GwAAAAC0E9u3b09NTU2bte9+FbQ/gj/gffPb3/42U6ZMyWmnnZZ+/frlzDPPzC233JK6uro93vaFF16YO+64Y4+3sycNHz48K1eubOtuAAAkSYYOHZrjjjsu/fr1a7FcdNFFbd01AIA9ZuPGjZk+fXpOP/30nHjiiTn11FMzZcqUvPzyy23SnyuuuCI/+MEPkiQvvvhiPvjBD+bFF19839p3vwraH8Ef8L745S9/mXPOOSeHHXZYfvCDH+TJJ5/M3Llz89RTT+Wiiy7K9u3b27qL7d769evbugsAAC3MmDEjTz75ZItl3rx5bd0tAIA95oorrsj69eszf/78/OpXv8oPfvCDbNu2Lf/zf/7PNDQ0vO/9aev7RW3dPrAzwR/wvrjmmmsyevTofPGLX8yBBx6YJPmLv/iLfP3rX89BBx2UmpqavPTSS7n88sszZMiQnHLKKbnyyivz6quvJkmWLVuWD37wgy2O+fd///f5+7//++bX99xzT84444wMGjQoV1xxRb7whS+0+JXf6tWrc9FFF+Xkk0/OsGHD8q//+q+71feGhoZMnz49p5xySgYNGpQLLrggTzzxRJLkYx/7WL797W+32P6ss87K/Pnz8/3vfz8XXHBBvva1r2XgwIEZPHhw/umf/inf+973csYZZ6R///655pprmvcbOnRovvnNb2bEiBHp169fxo4dm1WrViVJRowYkSQZP3585s6dmyT5t3/7t1RWVuakk07KiBEjUl1dncbGxua/zbXXXptLLrkk/fr1y/Dhw/PYY4/luuuuy8knn5xTTjkl999/f3Pbd9xxR04//fQMHDgw5557bv7P//k/u/W3AQDYlW3btuUb3/hGhg0bloEDB2b8+PFZvXp1kuQb3/hGPvWpT7XY/uabb87FF1+cJFm3bl0mTZqUU045JaeeemquueaabN68OcmbY8LTTz89V155ZQYMGJB/+Id/eH/fGACwz3viiScyfPjwHHzwwUmSioqKfPWrX80JJ5yQ119/PevXr8/VV1+dU089NYMGDcoll1yS3/3ud0n+3y/ybrzxxpx88smZMWNGkuRHP/pRzjrrrPTv3z+VlZX5+c9/vlt9ueqqq/If//EfufPOO3PppZc2r1+4cGE+9rGP5cQTT8y4cePyyiuvJEm+//3vp7KyMhdddFEGDBiQhQsXZuvWrbnpppty+umn5+STT86FF16Yp59+uvlYv/zlL/OZz3wmp556ao477rhUVlbmV7/6VZJd368C2p7gD9jjXnjhhaxcuTKf+MQndqpVVFRk9uzZOeyww3LRRRelQ4cOefjhh/PjH/84SXLppZfu1relfvSjH+Wb3/xmbr311vz85z/PgAED8vDDD7fYZsmSJbnyyiuzbNmyVFZW5itf+Urq6+vf9dgPPvhgnnzyyfz4xz/OL37xixYDs8rKyjz44IPN2/7617/Oiy++mI997GNJ3hwM9ujRI0uXLs0Xv/jF3HDDDVm2bFkWLVqU6urqzJ8/P48//njz/t/97ndz++2357HHHsvRRx+dSy+9NPX19fnJT36SJJk7d27Gjx+fpUuX5vLLL8/nPve5LF++PLfddlvuvvvu3HPPPc3HeuCBBzJ+/Pg88cQTOf744/PZz342Rx55ZB577LFccsklufbaa7Nt27YsXbo03/3ud3P//fdn2bJlOe+883LVVVft1t8GAGBXvv71r+eR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586dU1NTk6OPPrp5/QsvvJChQ4fmqaeeynXXXZf77rsvH/rQh5Ik8+bNy29/+9s90jfgz8NPOYD3xdVXX50HHngg3/zmN7N+/fo0NTXlP//zP3PppZfm2GOPzUUXXZTevXtn2rRp2bRpUzZt2pTp06fn8MMPz0knnZTDDz88HTt2zI9+9KMkyS9+8YssXbq0+fjnn39+vve97+Xpp59OQ0NDHnjggeYHDf+p/v3f/z2f//zn8+KLL6akpCRlZWXp2LFjunXrliT5q7/6q2zfvj1VVVU7feO9te6+++6sXr06b7zxRm644YYcddRR6devX5I3v+m+adOmJMm5556bn/70p/nxj3+c7du35ze/+U3mzp2bc889t9VtPvPMM/nc5z6XFStWZL/99stBBx2UJM2hKgBAaxQXF2fMmDG59dZb8/LLL6exsTELFizIJz7xiaxevTrJm988Hz16dGbMmJFjjz22+UbT8ccfnyOOOCI33nhj6urqsnXr1syaNSvjxo1r8U11AIC2cPLJJ+eggw7KV77ylTz33HOpr6/P5s2b88Mf/jC/+93v8tGPfjSnn356rr/++qxduzZbt27NLbfc0jyt+a6cf/75ueeee/L0008nefM+TWVlZR566KHd6tNb7xe1VnFxcc4999zcdtttWb16dbZt25bvfOc7WbVqVUaOHJlNmzaluLi4OYT81a9+lXvuuSfbtm37s7QP7Bl+8Qe8LwYOHJh//ud/zre//e2MHDkyb7zxRioqKvLRj340l1xySTp16pQ777wzN954Y0aMGJFt27blwx/+cO6+++507NgxhxxySL761a9m9uzZue666zJ48OBUVlbmjTfeSJKMGDEiL7zwQiZOnJht27bltNNOy4c+9KE/y3PqPvOZz+SVV17Jpz71qWzevDmHHXZYvv71r6dnz55J3rxxdfbZZ+eee+7JnDlz/qS2+vfvn8suuyxr1qzJySefnH/4h39o/rXjJz/5yVx55ZUZN25crrjiinzjG9/It771rXz1q19NeXl5Pv3pT7+nBymPGDEiv/vd7zJhwoSsX78+Bx10UL761a82Tz8BANBaU6ZMyR133JELLrggGzZsSK9evVJVVZVjjjmmeZvzzjsv8+bNy2WXXda8rmPHjrnzzjvzta99LX/913+dP/zhDzn++ONz9913p3Pnzm3xVgAAmpWUlOR//a//lW9+85uZMGFCamtr06lTp5x44om5++67c/TRR+emm27KLbfcknPOOSdbtmzJiSeemO985zspKyvL5s2bdzrmRz/60WzZsiVf/epXs2bNmpSVlWXcuHG58MILd6tPo0ePzvTp0/PrX/86N910U6vf09/93d/ljjvuyLhx47Jhw4Z88IMfzD/+4z/mL/7iL3LkkUfmggsuyNixY9PY2JgPfOADufDCC3Prrbdm3bp1qaio2Ol+FdD2ippMyAsUgBUrVqRbt2457LDDmtdVVlbmU5/6VM4///w93v4999yTRx99NHfdddd7PsbQoUPz+c9//k/+1SAAAAAAAPsmU30CBWHp0qW59NJLs3bt2jQ1NWXRokVZtWpVhgwZskfbXbt2bZ5++ul85zvfyac//ek92hYAAAAAAPwxpvoECsLf/M3f5KWXXso555yTurq6HHXUUZkzZ0569er1R/erra3NmWee+Ue3efLJJ9+x9sgjj+T666/PqFGjMmzYsPfUdwAAAAB4u8rKyvz2t799x/rcuXMzYMCA97FHwN7AVJ8AAAAAAABQAEz1CQAAAAAAAAVA8AcAAAAAAAAFQPAHAAAAAAAABUDwBwAAAAAAAAVA8AcAAAAAAAAFQPAHAAAAAAAABUDwBwAAAAAAAAVA8AcAAAAAAAAFQPAHAAAAAAAABeD/B7tMGOXIf04dAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 1800x600 with 3 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "cols = ['Cough_symptoms', 'Fever', 'Sore_throat']\n",
    "\n",
    "# Set up the figure and axes\n",
    "\n",
    "fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))\n",
    "\n",
    "# Plot univariate distributions for each column\n",
    "for i, col in enumerate(cols):\n",
    "    sns.countplot(data=df , x=col,hue='Corona', ax=axes[i]  , palette=\"Set2\")\n",
    "    axes[i].set_title(f'Distribution of {col}' , fontsize=18 )\n",
    "    axes[i].set_ylabel('Count')\n",
    "    axes[i].tick_params(axis='x')\n",
    "    axes[i].grid(True)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 123,
   "id": "3335dbb1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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HB0dmTdvHjk5OYwZM6ZUP56VdLvr1avHrbfeyn//+1/OnDmDn58ftra2dOjQgfXr1/PXX3+VS0CWv/6vv/66wPQFCxYwffr0Mq3bS9tIcnIyf//9NyEhIbRp0wY7uwuPl9iyZQvwvzt97du356effiIxMdGybFJSEqNGjeKbb74Brq39V3d6qIcIFx6p/tNPPwEXDj7Jycls27aNDz/8kIEDB3LXXXcVuVxAQABvv/02EyZMYODAgeTk5LBs2TJcXFzo2LEjcOGXrwMHDrBjx45Sv9MmISGBp59+mmHDhvHPP/8wf/58OnfubBms26NHD9577z1efPFFBg8eTGxsLCtWrChwwM1/ItWOHTto3rw5d9xxR4F1eHl5ce+997Jo0SKysrLw9/fn//7v/1i0aBH+/v507dq1VGW+mIuLC0888QSLFi3C3t6eXr16cfz4cV577TW8vLwIDg4ucV6dO3emf//+zJ8/n0OHDtGnTx/c3Nw4efIkq1at4uTJkyxcuBC48Ctnhw4dCAsLIywsjObNm/PLL7/w+uuv06VLF9zc3MjOzmbChAncdNNNPPHEEzg4OPDII4+wfPlyevfuXaieRKRq0PG+4h/vL+fWW2/l6aefZv78+bzyyitMmjSpRMuVZrvvvPNO3njjDW666SZL10R/f39mz55N/fr1Czw9s6zccsst3H333cybN4+srCxuu+02tm3bxubNm1m4cGGZ1m1RbaRRo0asXLmSBg0aUKdOHbZt22bpipvfK2X48OF8/PHHPP744zz11FPUqFGDJUuWUL9+fQYNGgT8r/3v2bOH9u3bFzu2Uv5HAZkI8McffzBkyBDgwi887u7u3HTTTcyePbvQINqLdevWjXnz5rFixQrLwG5fX1/effddyxiEoUOH8ttvvzFq1ChmzZpF/fr1S1yu+++/n6ysLMLDw3FwcGDAgAGMHTvWcpDr3Lkz48eP57333mPTpk3cdtttLFq0iAceeMCSh7OzM4899hgffvgh33//Pdu3by+0nsjISG688UbWrl3L8uXLqV+/PsOGDSM8PPyaf/F6+umn8fDw4P3332f16tW4uLhw9913M3r06EJ974szd+5c/P39+eSTT5g0aRKZmZm4ubnRuXNnZs2aZXkPmclk4q233uK1115jyZIlJCYm4unpyfDhwy1PkVq4cCF//fUX//3vfy1dYp555hk2bdrE+PHj+eSTTyxPUhORqkPH+8pxvL+cUaNG8cMPP/Dee+8REBBQ4ndelXS78wOy/LtqgOUu5Z133lluQcbcuXNZtGgR7733HsnJydx0000sXLjQ8nTOsqrbotrI4sWLiYyMZMKECTg4OODl5UV0dDQzZ85k7969DBs2jIYNG/LBBx8wd+5cXnjhBRwcHPDz82Pu3LmW9v/UU0+xePFiRo0axcaNG7nhhhvKo6qqJBvjSqP8REREREREpNzoDpmIiIiISBnLy8u74tMN8+WP25LqS3fIRERERETK2LBhwyzvMLuSQ4cOXYfSSEWmgExEREREpIz99ddfZGRkFDtf69atr0NppCJTQCYiIiIiImIlemGAiIiIiIiIlSggExERERERsRIFZCIiIiIiIlaigExERERERMRK9OKDcpCYmIYelVL2bGzA3b226lcqJbXf8pVfv3J5Vb3t6W+satJ+rXqq0z4t6blJAVk5MAyqfAOzJtWvVGZqv2It1aXtVZftrG60X6se7dP/UZdFERERERERK1FAJiIiIiIiYiUKyERERERERKxEY8hERC7DbDaTl5dbJnnZ2EBWVhY5OdnqM38VbG3tMJn0G2J5Kcu2bi1V5W9MbV2k+lFAJiJyCcMwOHs2iXPn0ss036QkE2azuUzzrE6cnJypU8cNGxsbaxelyiivtm4tVeVvTG1dpHpRQCYicon8C1RnZ1ccHGqU2UWRra0NeXmV+Kd7KzEMg+zs86SnJwNQt667lUtUdZRXW7eWyv43prYuUj0pIBMRuYjZnGe5QHV2rlOmedvZmcjNrfy/3luDg0MNANLTk6ld21VduspAebZ1a6kKf2Nq6yLVj/7KRUQukpeXB/zvokgqjvx9UtnHOlUUausVl9q6SPWigExEpAiVvetWVaR9Uj5UrxWP9olI9aKATERERERExEo0hkxEpII5duwo7767gn379pCeno6rqys9evTm0Ucfp2bNmtYunkiZUVsXEVFAViGZTDaYTOquIFId/frrz4wZE86DDw5jxYqVuLq6cuzYUV55JZIxY8JZvHgZtra21i6myDVTWxe5NrperDoUkFUwJpMNLq41sdVTlYpkNsyYTJX7scYiV/LKK5HcfXd/Hn/8Scu0pk1vZNq0WcydO5P4+BPY29uzeHEU+/fvxWQy4evbgfDw0Xh4eLB/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v63b0rRpM+6//yHWrv2InJyc69rWRaR6sTHyO09bScuWLXn33XctT1n84YcfeO6557j//vt5/vnnCxx0n3/+eZydnXn55Zct0/r27cuIESMYPHgw3bp1Y8KECZaukGfOnLF0jzSbzfTp04ft27dbui5s3LiROXPm8MMPP/DRRx/x9ttv88UXX1jynjJlCpmZmcydO7dU25SYmMbV1qqt7YWLjfnbP9MgzUs0quPGc537k5KSQW6u+vRXVPltuDI++QnAycaWto71qetaD1Mxd/JLw85ki3tNZ/LyzFd9fKjucnKySUz8Fw+PhtjbOxRIs7HRj2HFSUgoeG7Kr09398L1WVnZ2ZmqxPmhKu6bq2VjAx4etQu13+rOzq7yXi/mX88lJ1f967n89lucCnWH7KeffiI8PJypU6cSEhJSKD0kJITw8HD69u2Lr68vK1euJDExkcDAQACCg4OJjo6mdevWuLq6MnPmTPz8/Gja9ML7Q3x9fZk5cybTpk0jOTmZxYsXW9YTGBjI3LlziYmJYejQoezbt48NGzawePHiUm+HYaCDRjlS/VYOlfXJT3VsHchzcCfHnItNXtnnr7Z77XQMEBGRqqRCBWRvvvkmubm5REZGEhkZaZnu6+vLsmXLCAgIYMqUKUydOpVTp07h5eXF0qVLcXFxASA8PJzc3FyGDh1KRkYG/v7+LFy40JJPVFQU06ZNo1evXphMJgYNGkRYWBgArq6urFixgsjISKKionBzc2PSpEl07NjxelaBiIiIiIhUI1YPyA4dOmT5/5tvvlns/EFBQQQFBRWZZm9vT0REBBEREUWme3h4EBUVddm8W7duzapVq4otg4iIiIiISFmonKPuRUREREREqgAFZCIiIiIiIlaigExERERERMRKFJCJiIiIiIhYiQIyERERERERK7H6UxZFRCoLe1tb7Gyv7ncsO5MdtqVc1mw2MJv1wi25/kwmG0wmm+u2PrV1EanOFJCJiJSAva0t3o0aYGuyvW7rzDObSUnOrBQXqufPnyc1NYX69T2tXRS5RiaTDS6uNbE1Xb9ONGrrIlKdKSATESkBO1sTtiZbVv68lVPpqeW+Pk/nugy9oysmk02luEgNDx9FcPBg7rlnAD//fICIiGfYvHmrtYslV8FkssHWZFJbvwy1dREpawrIRERK4VR6KifOJlm7GBVOSkqy5f93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bl13atd2JS8vt/gFKjAbG3B1rUVyckalbiMXt3URqR4qVEDm6urKihUriIyMJCoqCjc3NyZNmkTHjh0BCAgIYMqUKUydOpVTp07h5eXF0qVLcXFxASA8PJzc3FyGDh1KRkYG/v7+LFy40JJ/VFQU06ZNo1evXphMJgYNGkRYWFiJ1i0i1U+uYSYtL7tM8qpt5OHo6Mi5c3m6wysVjslkwmRysHYxromNzYWx3/b2OZU6IBOR6sfqAdmhQ4cKfG/dujWrVq267PxBQUEEBQUVmWZvb09ERAQRERFFpnt4eBAVFXXZvItbt4iIiIiISFnSPXERERERERErUUAmIiIiIiJiJQrIRERERERErEQBmYiIiIiIiJUoIBMREREREbESBWQiIiIiIiJWooBMRERERETEShSQiYiIiIiIWIkCMhEREREREStRQCYiIiIiImIlCshERERERESsRAGZiIiIiIiIlSggExERERERsRIFZCIiIiIiIlaigExERERERMRKFJCJiIiIiIhYiQIyERERERERK1FAJiIiIiIiYiUKyERERERERKxEAZmIiIiIiIiVKCATERERERGxEgVkIiIiIiIiVqKATERERERExEoUkImIiIiIiFiJAjIRERERERErUUAmIiIiIiJiJQrIRERERERErEQBmYiIiIiIiJUoIBMREREREbESBWQiIiIiIiJWooBMRERERETEShSQiYiIiIiIWIkCMhEREREREStRQCYiIiIiImIlCshERERERESsRAGZiIiIiIiIlSggExERERERsRIFZCIiIiIiIlZSIQOy33//naFDh9K+fXu6dOnCjBkzyM7OBuDnn39m8ODB+Pj40LNnT1avXl1g2fXr1xMYGEjbtm0JDg7mwIEDlrS8vDzmzJlDp06d8PHxITQ0lNOnT1vSExMTCQsLo3379vj7+xMZGUlubu712WgREREREal2KlxAZjabefLJJ+nTpw+7d+9mzZo1bNu2jaVLl5KamsoTTzzBoEGD2LNnD5GRkcyaNYtffvkFgF27djF9+nRmz57Nnj17GDhwIKGhoZw7dw6A6Ohotm/fztq1a9m6dSuOjo5MmjTJsu7Ro0dTs2ZNtm7dypo1a9ixYwcxMTHWqAYREREREakGKlxAlpqaypkzZzCbzRiGAYDJZMLJyYlNmzbh4uLC0KFDsbOzIyAggAED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YiUKyERERERERKxEAZmIiIiIiIiVKCATERERERGxEgVkIiIiIiIiVqKATERERERExEoUkImIiIiIiFiJAjIRERERERErUUAmIiIiIiJiJQrIRERERERErEQBmYiIiIiIiJUoIBMREREREbESBWQiIiIiIiJWooBMRERERETESipsQJaXl8ewYcOYMGGCZdrPP//M4MGD8fHxoWfPnqxevbrAMuvXrycwMJC2bdsSHBzMgQMHCuQ3Z84cOnXqhI+PD6GhoZw+fdqSnpiYSFhYGO3bt8ff35/IyEhyc3PLf0NFRERERKTaqrAB2aJFi9i7d6/le2pqKk888QSDBg1iz549REZGMmvWLH755RcAdu3axfTp05k9ezZ79uxh4MCBhIaGcu7cOQCio6PZvn07a9euZevWrTg6OjJp0iRL/qNHj6ZmzZps3bqVNWvWsGPHDmJiYq7rNouIiIiISPVSIQOyHTt2sGnTJu666y7LtE2bNuHi4sLQoUOxs7MjICCAAQMGsHLlSgBWr15Nv3798PX1xd7enuHDh+Pq6srGjRst6aNGjaJhw4Y4OzszceJEtmzZQlxcHEePHmX37t2MHTsWJycnmjRpQlhYmCVvERERERGR8lDhArLExEQmTpzIq6++ipOTk2V6bGws3t7eBeb18vLi4MGDABw+fPiy6WlpaZw8ebJAuoeHB3Xr1uXQoUPExsbi4uKCp6enJb158+bEx8dz9uzZUm+Djc3Vf6R411K/+pT/R67M2vunKn9EREQqIztrF+BiZrOZsWPH8thjj3HLLbcUSMvIyCgQoAE4OjqSmZlZbHpGRgYANWvWLJSen3bpsvnfMzMzqVOnTqm2w929dqnml9Jxcall7SKIXDW1XxEREblYhQrIlixZgoODA8OGDSuU5uTkRFpaWoFpWVlZ1KpVy5KelZVVKN3V1dUSXOWPJ7t0ecMwCqXlf8/PvzQSE9MwjFIvBoCtrQlXV12wXUlKSga5uWZrF0MuQ234ytR+y4eNjX4MExGRyqlCBWSffPIJp0+fpn379gCWAOvrr79m3LhxbN++vcD8hw8fpkWLFgC0aNGC2NjYQundunWjbt26eHp6FujWeObMGVJSUvD29sZsNpOSkkJCQgIeHh4AHDlyhAYNGlC7dulP8IbBVQdkUjzVr1Rmar8iIiJysQo1huzLL79k//797N27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pjB16lROnTqFl5cXS5cuxcXFBYDw8HByc3MZOnQoGRkZ+Pv7s3DhQkv+UVFRTJs2jV69emEymRg0aBBhYWHXa9NFRERERKSaqXAB2WOPPcZjjz122fTWrVuzatWqy6YHBQURFBRUZJq9vT0RERFEREQUme7h4UFUVFTpCiwiIiIiInKVKlyXRRERERERkepCAZmIiIiIiIiVKCATERERERGxEgVkIiIiIiIiVqKATERERERExEoUkImIiIiIiFiJAjIRERERERErUUAmIiIiIiJiJQrIRERERERErEQBmYiIiIiIiJUoIBMREREREbESBWQiIiIiIiJWYmftAoiIiEjFYTLZYDLZWLsYIiLVhgIyERERAS4EYy6uNbE1Vc4ONGbDjMlkQ16eYe2iiIiUmAIyERERAS4EZLYmEyt/3sqp9FRrF6dUPJ3rMvSOrtjY2AAKyESk8lBAJiIiIgWcSk/lxNkkaxdDRKRaqJx9EkRERERERKoABWQiIiIiIiJWooBMRERERETEShSQiYiIiIiIWIkCMhEREREREStRQCYiIiIiImIlCshERERERESsRO8hExG5jmxt9TvYpcxmA7NZL/IVEZHqSQGZiMh1UNvBEcNspk4dJ2sXpcIxzGaSkjMVlImISLWkgExE5DpwsnfAxmQi9ev/kpd82trFqTBsXetTt/eDmEw2CshERKRaUkAmInId5SWfJjfhhLWLISIiIhWEAjKpdDQGpzCNwRERERGpnBSQSaWhMTiXpzE4IiIiIpWTAjKpNDQGp2gagyMiIiJSeSkgk0pHY3BEREREpKrQYBwREREREREr0R0yEREREanQKusDvfTQLSkJBWQiIiIiUiGZTDYYZjOurrWsXZSrooduSUkoIBMRERGRCsnGxqbSPtBLD926Mt31/B8FZCIiIiJSoemBXlVHZX+NUXnc9VRAJiIiIiIi10Vlfo1Red31VEAmIiIiUsWZTDaYTDbWLkapVdZubVI83fX8HwVkl0hMTGTy5Mns3r0bW1tbBg4cyPjx47GzU1WJiIh16Nwk18JkssHFtSa2JgU3IhWRjuSXGD16NJ6enmzdupWEhARCQ0OJiYlh5MiR1i6aiIhUUzo3lVxlvaNSno9HN5lssDWZWPnzVk6lp5bLOsrLLfVu4B7vdtYuhki5UkB2kaNHj7J79262bNmCk5MTTZo0ISwsjLlz5+qkJyIiVqFzU8noQQHFO5WeyomzSeWWf3moX6uOtYsgUu4UkF0kNjYWFxcXPD09LdOaN29OfHw8Z8+epU6dkh0UTCYwrvF42qiOGw622j0Xq+98of5tPW4AOwcrl6bisHXxsPy/IvVGURsuSO23aGXVfm0q39CYErPGuaky/v3e6OqBjclExv7vyMuoXHeBbGvVpVa7Hv9/jFf5BWSVcb9W5mPn9Tg/a59eX6XdpyU9N1WuPVjOMjIycHIq+Mta/vfMzMwSn/Tc3Gpfc1mGtO50zXlUVXV7DLZ2ESqkivbSTLXhoqn9Fq2itd+KxBrnpsr891urXQ9rF+GqlfffQWXer5X52Fme+1X71DrKep9WoN/Tra9mzZqcO3euwLT877Vq6WJBRESuP52bRESqNgVkF2nRogUpKSkkJCRYph05coQGDRpQu/a13/USEREpLZ2bRESqNgVkF2nWrBm+vr7MnDmT9PR04uLiWLx4MSEhIdYumoiIVFM6N4mIVG02hnGtj5+oWhISEpg2bRq7du3CZDIxaNAgIiIisLW1tXbRRESkmtK5SUSk6lJAJiIiIiIiYiXqsigiIiIiImIlCshERERERESsRAGZiIiIiIiIlSggExEREalG/vnnnzLJJy8vj7i4uDLJK9/58+c5efJkmeYp1nP69GkyMzOtXYwKTwGZlKuNGzcSEBCAr68v33333XVZ5/Hjx2nZsiXHjx+/LusTuRY9e/Zk3bp11i6GiFWtXLmSli1bEhMTU2C6tf8+1q1bR8+ePa/rOv/++2/Gjx9Pt27d8PHxoXfv3sybN4+MjIwyyf+PP/6gf//+ZZLXmDFj+Pjjj68pj7179+Lj42P5/tBDD/Hjjz8CsGvXLlq2bHlN+ZeVmJgYOnToQIcOHdi9e3eBMl9vEyZMYMKECVZbf77ff/+dZ555ho4dO+Lj40NgYCBz5swhJSUFuPB02D59+pCUlATA66+/zrBhw6xY4opLAZmUq9WrV9OvXz/27dtHjx49rF0cERGpgFauXMmDDz7Iu+++S25urrWLYzX79+/n3nvvpVGjRnz88cccOHCApUuX8vPPPzNixAjy8vKueR1paWnk5OSUQWkhOTn5mvNo3749Bw4cKNM8y8MHH3xAWFgYe/bswc/Pr0CZq6PvvvuOhx56iJtuuolPPvmE/fv38+abbxIXF8egQYM4deoUWVlZujtWQgrIpNyEhISwc+dOVq1aRe/evTl27BhPPfUU/v7+9OjRgwULFpCdnQ1c+BXyoYceYs6cOfj5+dGxY0fee+89PvroI3r06IGvry8vvfSSJe8jR47w5JNPcuedd9KmTRvuueeey96BS0hIICIigs6dO9OlSxdeeukl0tPTr0sdSNWUfxf2448/pkePHrRt25YXXniBvXv3MnDgQHx8fHj00UdJSkoiPT2dSZMmcdddd9G2bVu6du3Km2++WWS+2dnZvPbaa/Tq1Qs/Pz9GjRrF0aNHr/PWiVxfO3bsIDExkQkTJmA2m/nqq68KpP/+++8EBwfj5+fH448/bulul/93OHv2bDp06MDLL78M/O+HwHbt2jFgwAA+/fRTS16nTp1i9OjR9OzZkzvuuINevXqxZs0aS/qRI0cYNmwYPj4+DBgwgD/++KP8K+AiL730EoMGDeKZZ57Bzc0NgJtuuokFCxbg7u5u6R544sQJRo8eTUBAAJ07d+b555/n9OnTwIW7Sj179iQ6OpquXbvi5+fH008/bXmp+KhRowDw8fHhwIEDxR6jkpKSiIiIoEOHDvj7+zNmzBhSU1OZOHEie/fuZcmSJTz11FMFtsNsNtOpUye+/vpry7SePXsyevRoy/c5c+Ywbty4AnfBRowYQXx8PFOmTGHatGmWeZcvX05gYCBt27blmWeeuew5PDc3l9dee43u3bvTrl07hg4dysGDBwHIysrilVdeoXv37nTo0IFhw4bxyy+/WJZt2bIl7733Hn369MHHx4cHHniAQ4cOAdC5c2eOHTvG/PnzeeSRRwqUuah2+Prrr/Pss88yfvx42rVrR7du3fjiiy9444036NSpE35+fixevNiy7uKuU7755hv69etH27ZtefLJJ60etGZnZzNp0iSefPJJxowZg6enJzY2NjRv3pyoqCgaNGhAZGSk5U5s//792bhxIwAZGRlMmjSJLl264O/vz4IFCwrke6VzYMuWLZkxYwb+/v6F2lylZ4iUo4cfftiIiooyMjIyjB49ehjz5s0zsrKyjPj4eCMkJMSYN2+eYRiGsXbtWsPb29t4++23jby8PGPlypVGq1atjOeee87IzMw0fvnlF6NVq1bG7t27DcMwjL59+xrz5s0zsrOzjfPnzxuRkZFGt27dDMMwjLi4OMPb29uIi4sz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80qCg4NZvnw5f//9N7m5uSxZsoSvv/4aNzc37rvvPubPn8/Ro0fJzs7mnXfe4fDhw/Tr1++q1lVWirtOGThwIH/++ScfffQRubm5bNu2jc2bN1u1zPb29syaNYv333+fBQsWcOrUKcxmM4cPH+Y///kPJ0+e5IUXXrC0tZI8SK26nwMVkMl14ezsTExMDLt27aJbt2707t0bk8lEdHR0qfO6+eabGTduHGPHjsXX15dnn32W++67D3t7e/78888C89rZ2bFkyRISEhK466676NKlC8eOHePtt9+26q9LUn3MmjWLjRs30q5dO4KDg/H09OTWW28t1FYBxo8fzx133MFDDz1E+/btiYmJISoqiltvvdUKJRcpXx988AEDBgzA3t6+UNqQIUP45JNPSExMpHfv3jz11FN069aN1NRUFi9eXKBL08X69u3LCy+8wIwZM2jfvj1Tp05l3LhxDBo0iJo1azJz5kzeeOMNfHx8eOSRR+jcuTMeHh78+eef2Nra8tZbb3H69Gk6derEyJEj6dWrV3lXQwF+fn68//77/PHHH5YnRea/52nZsmXY29tbzmu5ubn06dOHHj16kJOTw9tvv42dXfEdn7y9vfH19aVr16788MMPxR6j5s2bh7OzM3379qVXr164ublZnmg5aNAg1q5dy0MPPVTkuu68807S09MtY8WaNGlCgwYN8Pf3L9Dl8mIhISEsWLCg0J2Xkhg5ciQDBgzg8ccfx9/fn71797J06VLs7e0ZN24cXbp0Yfjw4fj7+/PFF1+wfPlybrrpplKvpywVd53SpEkT3nzzTVauXImvry+LFy8mMDDQqmUG6Nq1K6tWreLYsWPcd999tGvXjqeeeoqmTZvy6aef0rBhQzw8PAgMDGTIkCH897//LTbP6nwOtDFKe29bREREREREyoTukImIiIiIiFiJAjIRERERERErUUAmIiIiIiJiJQrIRERERERErEQBmYiIiIiIiJUoIBMREREREbESBWQiIiIiIiJWooBMRERERKq9f/75p0zyOX/+PCdPniyTvKR6UEAmUg2kpqYydepUunfvTtu2benSpQvjx4/XCUNEpJpo2bIlu3btKjBt7dq1tG7dmg8++MBKpao4vv32Wx5//PEyyeuhhx7ixx9/LJO8pHpQQCZSDYwZM4bk5GTWrFnDTz/9xMcff0x2djaPPfYYubm51i6eiIhcZ2+99RYzZswgKiqKhx56yNrFsbqUlBQMwyiTvJKTk8skH6k+FJCJVAP79u0jMDCQevXqAeDh4cGLL77IHXfcwdmzZ0lPT2fatGl0796dgIAAxowZQ0JCAgCff/45t99+OwcPHgTgjz/+oE2bNmzZssVq2yMiIlfHMAxmzJjBO++8w7vvvkuPHj0saS1btuS9996jT58++Pj48MADD3Do0CFL+t69exk6dCjt27enZ8+eLFy4kOzsbP78809atWplCUSOHz9Oy5YtC9x5Cw4OZs2aNbz++us888wzRERE0L59e7p168arr75a4vLHxcXx1FNP4evrS0BAAFOnTiU7OxuAQ4cOMWrUKPz8/OjWrRtTp04lLS0NgHXr1vHggw8yY8YMOnbsSEBAABMnTiQnJ4ddu3YxZcoU4uPj8fHx4dSpU5w6dYrRo0fTs2dP7rjjDnr16sWaNWuKLceIESOIj49nypQpTJs27ep2klQ7CshEqoF+/foxZcoUpk6dysaNGzlx4gT16tVj9uzZuLm58eKLL3L06FHWrVvH119/jbOzM//5z38wDIN+/foxYMAAxo0bR2pqKmPGjGH48OF069bN2pslIiKlkJOTw/PPP8/q1atZtWoVrVu3LjTP559/zvvvv8+WLVtwcnLilVdeAeCvv/7iscce46677uLHH3/k7bff5ttvv+WVV17B29ubJk2asH37dgC2bduGo6OjpdveqVOnOHToED179gRg06ZNdOnShV27djF9+nSWLl3KTz/9VGz5c3Nzefzxx6lXrx5btmzhs88+46effuL1118nOTmZRx55BC8vL7Zs2cLatWv5+++/GTdunGX5/fv34+7uztatW1myZAkbN25k06ZN+Pv78/LLL3PDDTdw4MABPD09mTRpEvb29nz++efs37+fhx9+mOnTp5ORkXHFcqxYsYIbbriBl19+mZdeeulad5lUEwrIRKqBGTNm8NJLL/Hvv//y0ksv0bNnTwIDA/n0009JTEzkq6++YuLEibi7u1OrVi1efPFFfv31V37//XcAJk+eTHZ2Nvfeey/16tXj2WeftfIWiYhIaU2ePJmTJ09So0YN1q1bV+Q8w4YNo169etSuXZu+fftaHnSxYcMGWrZsyaOPPoqDgwM33nijJbgzm8306tXL0nNi27ZtPPDAA+zatYu8vDy+/fZbfH19cXNzA6BZs2YMGjQIW1tbunfvTr169Ur0QI39+/dz4sQJXnzxRWrVqoW7uzuLFi1i8ODBfPPNN9jb2xMREYGjoyP16tVj8uTJfPvtt5w5cwYAR0dHnnrqKezt7WnTpg0tW7bk77//LnJdM2bMYMqUKdjb2xMfH0+tWrXIysoiNTX1iuUQuRp21i6AiJQ/k8lEUFAQQUFBGIbBkSNH+OSTTxg3bhzPPfccAPfff3+BZWxtbTl+/Di33347NWvW5L777mPevHmEh4dja2trjc0QEZFrcMcdd/DKK6/w448/EhoaSqtWrbjrrrsKzOPh4WH5v52dnWVcVWJiIk2aNCkwb+PGjcnKyiIxMZHAwED+85//kJ2dzc6dO/n888/5+OOP+f333/nmm2/o06ePZbn87vP57O3tMZvNxZb/zJkzuLq64uTkVKAMAF988QU33HBDgfNTftqJEycAcHd3x8bGpsB6LzduLC4ujldeeYV//vmHZs2aceONNwJgNpuvWA6Rq6E7ZCJV3NatW/Hx8SElJQUAGxsbvLy8eP7557n11lstfe+/+OIL9u7da/msW7fOMrbg2LFjREdHM3jwYF555RU9nVFEpBJ68MEHcXBw4M477+SJJ55g/PjxBcaIXUmjRo04duxYgWnHjh3DwcGBunXr0rZtW0wmE6tWraJhw4Z4enrSqVMnvv76a/bs2UPv3r2vufwNGjQgOTmZc+fOWabt3buXmJgYGjVqRHx8PHl5eQXKB4UDwOLk5OTw5JNPEhQUxK5du/joo4949NFHS1QOkauhgEykiuvQoQPu7u688MILHDp0iJycHNLT0/n000/5559/6Nu3L3feeSeRkZEkJyeTk5NDdHQ0ISEhnD17lpycHJ577jn69evHjBkz6NChA2PHji3Rr5kiIlIxPfvss7Rt25bw8HDLD3ZX0q9fP44cOcI777xDdnY2x44dY/78+QwYMAAHBwdMJhM9e/Zk8eLFdOnSBYDOnTvz7rvvcsstt+Dp6XnNZW7Tpg3NmjVjzpw5nDt3joSEBGbNmkVSUhLdu3cHYN68eWRlZXHmzBkiIyPp2LEjjRo1KjbvGjVqcO7cOXJzc8nJySErKwtHR0dsbGyIj49n7ty5wIVg7UrlAHBwcLA8TESkJBSQiVRxjo6OfPDBB9SrV4/Q0FDat2/PnXfeyaeffsrbb79N8+bNeeWVV6hTpw6DBg2iY8eO/PDDDyxbtox69erx2muvkZyczIQJEwCYNm0ahw8fZsmSJVbeMhERuVomk4lXX32V3NxcRo8eXeDOUlEaN27MsmXL+Oqrr+jUqRMPPfQQnTt3LvDgit69e5OcnEznzp0B6Nq1K+fOnSMwMLBMymxvb8+bb77JqVOnuPPOOwkKCqJDhw4888wz1K5dm7fffps///yT7t27079/fxo1asRrr71Worzzf7zs0KEDcXFxzJw5kzfeeAMfHx8eeeQROnfujIeHB3/++ecVywEQEhLCggULiIiIKJPtlqrPxiirly6IiIiIiIhIqegOmYiIiIiIiJXoKYsiIiIiYnXBwcGXfQw9wNKlS2nfvv11LJHI9aEuiyIiIiIiIlaiLosiIiIiIiJWooBMRERERETEShSQiYiIiIiIWIkCMhEREREREStRQCYiIiIiImIlCshERERERESsRAGZiIiIiIiIlSggExERERERsRIFZCIiIiIiIlby/wCw6mU11Wz9fQAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 1000x400 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "cols = ['Sex',\t'Known_contact']\n",
    "\n",
    "# Set up the figure and axes\n",
    "\n",
    "fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))\n",
    "\n",
    "# Plot univariate distributions for each column\n",
    "for i, col in enumerate(cols):\n",
    "    sns.countplot(data=df , x=col,hue='Corona', ax=axes[i]  , palette=\"Set2\")\n",
    "    axes[i].set_title(f'Distribution of {col}' , fontsize=12 )\n",
    "    axes[i].set_ylabel('Count')\n",
    "    axes[i].tick_params(axis='x')\n",
    "    axes[i].grid(True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "61c3d7a5",
   "metadata": {},
   "outputs": [],
   "source": [
    "Cough_symptoms: A significant number of individuals who tested negative also reported no cough symptoms. However, among those who reported having a cough, the number of positive cases is relatively higher.\n",
    "\n",
    "Fever: Most individuals did not report having a fever. Among those who did, the number of positive cases is higher than negative cases.\n",
    "\n",
    "Sore_throat: The majority did not experience this symptom.\n",
    "\n",
    "Shortness_of_breath: This symptom was also not common among the individuals.\n",
    "\n",
    "Headache: A notable number of individuals reported not having headaches.\n",
    "\n",
    "Age_60_above: A considerable number of individuals were below 60 years of age.\n",
    "\n",
    "Sex: The number of females and males in the dataset is almost evenly distributed.\n",
    "\n",
    "Known_contact: Most individuals did not have a known contact with a confirmed case."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "id": "73ef8713",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAlQAAAGvCAYAAACdJUtuAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAAB/xElEQVR4nO3de1hVdd7//ycgylZU0ihsrBTPCSYHJY95l1YOnqWyFC0lvfOcqakd1Lu0RhhNcUrCSdMsKcFGJzxMIZWnraAbtJKvZVkqHlAUtiKwYf/+8McaCVDQrSK9HtfVdd3uvfZnrdZ4zbzvtT7v19vJbrfbEREREZFr5nyrL0BERETkdqeCSkREROQ6qaASERERuU4qqERERESukwoqERERkeukgkpERETkOqmgEhEREblOKqhERERErpMKKhGR29StzmW+1ecXqUxUUInIVUVGRtKiRQuHrDVt2jQeeeSRCv+uRYsWREZGGn8ODQ2lRYsWxj8tW7bEz8+PAQMGsHLlSgoKChxyvRW9xqv9c/m/w7XKysrilVdeISkpyQFXDY888gjTpk2r0G+Sk5MZNWqUQ84vUhVUu9UXICJyrR544AFmzpwJQEFBAefOneObb75h7ty5JCcns2DBApycnG7a9cTExBT789NPP01ISAhPPvmk8ZmXl9d1n+fHH3/kiy++YMCAAde91rX6/PPP+emnn27Z+UUqGxVUInLbcnd3p23btsU+e+SRR2jcuDFvv/02jzzyCH369Llp1/PHa4FLBVRpn4tI1aJXfiJSYXFxcTzwwAOkpKTw9NNP4+vrS7du3YiOji523Llz55g+fTpBQUG0a9eO8PBwCgsLb/j1hYaGctddd7F69epSv8/NzSUwMJC5c+cW+7ywsJDOnTsze/ZsAL7//nuGDRtGQEAAfn5+PPfcc6SkpFz39X3++ecEBwfj4+NDt27diIyMxGazGd+fOXOGyZMn06lTJ3x9fenbty9ffPEFAGazmaFDhwIwdOhQQkNDK3TuAwcO8Pzzz+Pn58f//M//sG7duhLHnDlzhtmzZ/M///M/+Pj40L59e8aMGcORI0eAS69t165dy9GjR2nRogVxcXHApfs6b948Hn74YXx8fOjduzfx8fHXcotEbjt6QiUi16SwsJCJEyfy3HPPMXHiRNasWUNERAQtW7akS5cuFBYWEhYWxpEjR5g8eTL169dn6dKlpKamctddd93Qa3NxcaFDhw7Ex8djs9moVq34f9XVqFGDxx9/nA0bNjBt2jScnS/9/5Zms5lTp07Rt29frFYrYWFhBAUFsWjRIvLz83n//fcZMWIEW7ZsoXbt2td0bVFRUSxYsIAhQ4Ywffp0fvzxRyIjI0lPTzcKvClTpnD69Glmz55NrVq1WLduHa+88goNGjSgdevWvPHGG/zf//0fb7zxBkFBQeU+94kTJxgyZAj33Xcf4eHhWK1WIiIiOH36tHGM3W5n1KhRnDt3jpdffhlPT09+/PFHFi5cyBtvvMGHH37I6NGjOXPmDD/88AOLFy/mvvvuw263M2bMGPbs2cP48eNp0qQJ//nPf3jppZfIy8ujX79+13S/RG4XKqhE5JrY7XZGjx5t7A8KCAjgP//5D4mJiXTp0oVvv/2W1NRUoqKi6NatGwAPPfTQNW1IvxZ33nkn+fn5nD17ljvvvLPE93379mXNmjUkJSXRvn17ANavX8/9999P27ZtsVgsnDlzhtDQUAICAgDw9vZm9erVWK3WayqosrOzef/993n66ad57bXXAOjcuTMeHh689tprPP/88zRr1oxdu3YxevRounfvDkBQUBAeHh64uLjg7u5O06ZNAWjatKnxf5fH8uXLsdlsREdHU79+fQAaN27MU089ZRxz8uRJTCYTr7zyCoGBgcb5jxw5Yjzxu++++6hXrx7Vq1c3Xmdu27aN7777jgULFvDXv/4VgC5dupCTk0NERAS9evUqUdiKVCX62y0i18zPz8/4v6tXr069evW4cOECAElJSbi6utK1a1fjmJo1a/Lwww+ze/fum3aNZW1Kb9euHX/5y1/48ssvad++PXl5efznP/9h2LBhADRr1ox69erx4osv0rNnTx5++GE6dOjA1KlTr/la9u7dS05ODo888kixV3xFRea2bdto1qwZQUFBREZGcuDAAR5++GG6du3KK6+8cs3nLZKcnEzbtm2NYgrgwQcf5J577jH+fPfdd7NixQoAjh07xuHDh/n555/Zs2cP+fn5Za69Y8cOnJycePjhh0v8u61bt46DBw/SqlWr6/53EKmsVFCJyDVzc3Mr9mdnZ2cjm+jcuXN4eHgYr9OKeHp63pRrO3HiBG5ubnh4eJT6vZOTE7179+azzz7j9ddf59tvvyUrK4u+ffsCUKtWLVatWsX7779PfHw8q1evxmQy0adPH1599VVq1KhR4Ws6e/YsACNHjiz1+5MnTwKwYMEClixZwoYNG9i4cSPOzs507NiRWbNmce+991b4vEXOnTtHw4YNS3z+x/9M1q1bx/z580lPT8fDw4OWLVuW+M/6j86ePYvdbsff37/U70+ePKmCSqo0FVQickPccccdZGZmUlBQgIuLi/F5UVFxIxUUFLBr1y78/f2LnfuP+vbty5IlSzCbzfz73//G39+/WMHi7e1NeHg4BQUFpKam8q9//YtPP/2Uhg0bllkUXUmdOnUAiIiIoFGjRiW+L3o1Wbt2baZMmcKUKVM4dOgQX3/9Ne+99x6zZ89m6dKlFT5vkTvuuIOMjIwSn1/+n0lSUhKvvPIKQ4YMYcSIEUbMw7x580hOTi5z7dq1a1OzZk3j6dYf3X///dd83SK3A3X5icgN0aFDB2w2G1999ZXxWV5eHtu2bbvh5169ejUnT57kmWeeueJx3t7e+Pr68uWXX5KYmGg8nQLYuHEjDz30EKdOncLFxQU/Pz9mzZpFnTp1OH78+DVd14MPPoirqysnTpzA19fX+MfV1ZW///3vHDlyhKNHj/Lwww+zceNG4xpfeOEFOnbsaJz3SkXilTz00EPs3buXEydOGJ/99NNP/P7778af9+7dS2FhIePHjzeKqYKCArZv3w5gdGn+8clj+/btuXDhAna7vdi/28GDB/nHP/5R7DWgSFWkJ1QickN06NCBzp0789prr3H69Gn+8pe/sGLFCs6cOVNsD8/1sFqtWCwW4NL/0GdmZrJ161ZiYmLo06cPjz322FXX6NevH3PnzsXZ2ZmePXsan/v7+1NYWMiYMWMYOXIktWrVYsOGDWRnZ5dr3dLccccdhIWFsXDhQqxWK0FBQZw4cYKFCxfi5OREy5YtqV27Nl5eXrz11ltYrVbuu+8+9u/fzzfffGMkkxdtiE9MTKRu3bq0bNmyXOcfNmwYa9asYcSIEYwbN46CggLeffddXF1djWPatGkDwP/93/8xcOBAsrKy+Pjjjzlw4AAAFy5cwN3dnTp16pCRkcE333xDq1atePjhh2nXrh2jR49m9OjRNGnShNTUVCIjI+ncuTP16tW7pnsmcrtQQSUiN8zixYuJiIhg0aJF5Obm8te//pWnnnqKr7/+2iHr//DDDzz99NPApScm9evXp3Hjxrzzzjv07t27XGv89a9/5Z133qFbt27UrVvX+Pyuu+5i6dKlLFy4kFdffZWcnByaNWtGZGQkDz300DVf88SJE/H09OSTTz5h6dKl1K1blw4dOjBp0iSjUFq8eDHz589n4cKFZGZm0qBBA8aOHWu8ZmzWrBm9evVi1apVfPfdd/z73/8u17nvuOMOPv30U+bMmcO0adOoVasWYWFhxbKigoKCeOONN1i2bBkbN27kzjvvJCgoiMWLFzNmzBiSk5N5+OGHGTBgAN988w1jxoxh/PjxjBw5kg8++ICFCxcSFRXF6dOnufvuu3nuuecYM2bMNd8vkduFk13TLUVERESui55QicgtU559Nc7OziX260hxuo8it54KKhG5JY4cOcKjjz561eP69+/PO++8cxOu6Pak+yhSOeiVn4jcEnl5eaSlpV31uDvuuKPU7CS5RPdRpHJQQSUiIiJynfRCXUREROQ6qaASERERuU4qqERERESukwoqERERkeuk2ISb7PTpbNQGICIicntwcoL69Wtf9TgVVDeZ3Y4KKhERkSpGr/xERERErpMKKhEREZHrpIJKRERE5DqpoBIRERG5TiqoRERERK6TCioRERGR66SCSkREROQ6qaASERERuU4K9hQREZFKpd3fv72u3+9+uauDrqT89IRKREREKo3rLaYctUZFVbig2rFjB08++ST+/v506tSJN998k4sXLwKQkpLCk08+iZ+fH4888giff/75Fde6cOEC06dPJygoiICAAKZOncr58+dLPa5nz55ERkZecb3MzEymTZtGp06daNeuHcOGDePHH380vj99+jSjR48mMDCQoKAg5syZg81mM77ftGkTffv2xd/fn0ceeYTFixdTWFhofL927Vp69OhB27ZtGTBgAHv37i3XPRMREZGrc2QhdLOLqgoVVGfOnGHUqFE888wzJCUlsXbtWnbt2sUHH3zAuXPnGDlyJP369WP37t3MmTOHt99+m9TU1DLXe/PNN0lPT2fTpk1s3ryZ9PR0IiIiShw3e/Zsfv3116te36uvvkpmZib//ve/2bZtG/7+/oSFhXHhwgUAJk6cSM2aNfnuu+9Ys2YNO3bsYPny5QDs37+fqVOnMnHiRJKSkoiOjiYuLs743mw28+abb/LOO++we/du+vTpw4svvkhOTk5FbqGIiIiU4kYUQDezqKpQ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zf3334+3tze9evWioKCAmTNnlvp7xSbcGuvU4SciIuJwDotN6NGjB9OnT2fWrFl069aNrKwsHn30Ufr27Vv2yRWbcF0UlyAiIlI5OCw2AWDQoEGcOnWK33//naSkJDp06FDmOopNEBERkarCYbEJADt37uTLL78s8zXf5RSbICIiIlWFw2ITTp8+zYwZM4iIiChX55tiE0RERKSqcEhsgt1uZ+rUqYSGhuLj41PqMYpNEBERkarKIV1+9957L7t27WL37t387W9/w263A/D666+zadMmoqKiSo1NqFatGpMmTcJisWCz2fDz8ysWm7Bw4UJcXV05cuQIdevWpXXr1leMTUhPT2fQoEGcPHkSk8lETEwMjRs3plWrVkZswogRI0hJScHFxQVfX99isQnz588nOjoaZ2dnzp8/z8qVK40uv2bNmrF582Y++ugjTp06hbe3N5mZmbc0NmH3y13LvTFdG9JFRERuHId0+d1zzz0sXLgQNzc31qxZQ1paGp6engA8//zzpa5hMplo0KABe/bsYfXq1cTExGCxWGjYsCFubm64u7vzyy+/0LZtW5KSkpg1axYJCQn4+vqWeV0hISFYrVY2bNjAzp078ff3JywsjAsXLtCoUSPc3d05dOgQGzduJDIyErPZjLe3NwBxcXFERUUxePBg9u3bx9KlS4mLi2P58uUAtG7dGrPZTGhoKNu3b8fLy4v09HS6dOlyLbfQIVRMiYiIVA4VLqjWrl3L5MmTeemll4p9/sgjj5CQkICPjw82m82Y4VenTp1S18nJySE9PZ3AwECGDBnCs88+S0BAAMeOHSMnJ4cuXbrw4YcfcvDgQdq3b8/MmTOpXr06vXv3LnW9/fv3k5GRQW5uLv379ycoKIhly5aRkZFBfHw8hw8fxmq10rx5c4KDg5kwYQIdO3bkl19+AWDVqlUAxMbGEhAQQEhICCdOnCA6OhqAH374gaCgID755BM6derEqVOnaNCgAdu2bavoLXSIikQmVDReQURERCqmwq/8OnfuTO/evalWrVqJ2ISiJ0BFoZ7PP/98sa67yx0+fBibzUZ4eLixT6ko5uDXX3+lVatWdOrUiYCAAAIDA7FarfTs2ZNu3bqVup6Pjw9paWnFPluzZg1z5swhODiYbdu24eHhQVRUlPF9Wloaffr0ISsri9jY2GK/vXjxIn379jUKuJ9++omBAwcSGhpqHDNu3Dh1+YmIiIjjuvyK3HvvvaSkpLBmzRq+/PJLPvjgg1KPU5efiIiIVBUOG45cxNXVFVdXV3x9fRk6dKjRlacuPxEREamqHDYcefny5VgsFt59913js8uHGWs4soiIiFRVTvaijIMKOnPmDB06dGD69Ok899xz7N+/n6eeeoq7776bs2fPUrNmTaxWK3Pnzi32BOlykyZNwmw2FxuO3KFDByIiIkhPT+eJJ56gdu3aJYYjjxw5stT1RowYYQxHttlsxYYjAzz11FMcP368xHDkCRMmYLFYePbZZ/H09CQ7OxsPD49iw5F37NjBqFGjqFu3brHhyAkJCXh4eJT7vmVkZHNtd7wkdfmJiIjcWE5OcOedta96nMOGI8Ol13NFg4Nr1aqFyWTixIkTZZ9cw5GvmYopERGRyuO6YxOKnv4cPXqUwYMHk5CQwJ49e9i8eTN9+/Zl9+7dpa5TNBx54cKF7Nixg6SkJN577z3WrVtHTk6OMRw5Pj6e3bt38/e//x1XV9erDkeOi4vDYrFgsVgwm80AxYYjx8TEYDabMZvNzJw503ilOHToUJ577jlSU1PZu3cvKSkpDBs2jMaNGwP/HY6cmJiIxWIhMTERT09P4uPjK3oLr5siE0RERCoXhw1Hvtpw4T/ScGQRERGpKhwemwClxw6UdgwoNkFERERufw6PTSgrdkCxCSIiIlJVOSw2Aa4cO6DYBBEREamqrjk2AaBFixbcddddREREUKNGDYYMGcKsWbNo0qQJQ4cOZd++fVf8vWITrp26/ERERG68GxqbAJeiEwBOnjwJwJIlS7DZbMyePZtBgwaRl5dnvN4r8+SKTbgm6twTERGpXK6poCqKTliwYIHx2ZIlS+jfv7/xVAguveZbunRpqWsoNkFERESqimvaQ9W5c2d69+5NtWrVjDwqgAkTJuDl5WUUMldytdiEVq1aGbEJgYGB5Ofn07Nnz6vGJlzu8tiEbdu2XTU24fHHHze+K4pN6N27N3Bpv9TAgQOLra/YBBEREYFrfEJVVnSCl5dXuddQbIKIiIhUFQ6PTSiLYhNERESkqnJobMKVKDbBcXa/3LVcG9PV4SciInJz3LQnVH9kMpno2bMnERERnDlzhjNnzhAREUGvXr1wc3Ojbdu2/L//9/9YtmwZBQUFHDx4kH/84x8888wzZa750ksvcfz4cWJjY4sVUwCNGjUiICCAuXPnYrVa+f3333nvvfcICQkBwGKxMGbMGKZPn84rr7xS4pVmSEgI69evZ+fOneTn57N8+XJOnz5Njx49HH9zyuFqxZKKKRERkZvnunKo4FIW1YoVKwgKCjI+M5vNDB06lLS0tCv+1mq18re//Y2EhATy8/N59NFHef31143XgXv27GHevHkcPHiQO++8k2eeeYahQ4caEQaX+/777xkwYADVq1cvUQxFR0cTGBhIRkYG//d//4fZbMbZ2Zl+/foxefJkXFxc+N///V8SExNL7JMKCAgwOhX/9a9/8f7773PixAmaNm3Ka6+9ZjxdKy9H5VAVKe1JlYopERERxyhvDtV1F1RSMY4uqEREROTGueHBniIiIiJyiQoqERERkeukgkpERETkOqmgEhEREblOKqhERERErpMKKhEREZHrpIJKRERE5DqpoBIRERG5TiqoRERERK7TTRuOLJc4Od3qKxAREZHyKu//bmv0jIiIiMh10is/ERERkeukgkpERETkOqmgEhEREblOKqhERERErpMKKhEREZHrpIJKRERE5DqpoBIRERG5TiqoRERERK6TCioRERGR66SCSkREROQ6qaC6jZ0+fZrRo0cTGBhIUFAQc+bMwWaz3erLuuni4+N54IEH8PPzM/6ZMmUKACkpKTz55JP4+fnxyCOP8Pnnnxf77dq1a+nRowdt27ZlwIAB7N271/iuoKCAv/3tb3Ts2BE/Pz9efPFFTp48aXxfFe7/mTNn6NGjB2az2fjsVt6zq527sint/s2cORMfH59ifx9jYmKM73X/Ljlw4ADPP/887du3p1OnTkydOpUzZ84A+jtYXle6h/p7eAvY5bY1ZMgQ+8svv2y/cOGC/bfffrMHBwfbo6Ojb/Vl3XTvvPOOfdq0aSU+P3v2rL19+/b2jz/+2J6fn2/fvn273c/Pz56SkmK32+32nTt32v38/OxJSUn2vLw8+7Jly+xBQUH2Cxcu2O12uz0yMtLeu3dv+7Fjx+zZ2dn2iRMn2l944QVj/dv9/iclJdm7d+9ub968uX3nzp12u/3W3rOrnbuyKe3+2e12e//+/e1xcXGl/kb375KcnBx7p06d7AsXLrTn5ubaz5w5Y3/hhRfso0aN0t/BcrrSPbTb9ffwVlBBdZv69ddf7c2bN7cfP37c+OzLL7+0d+vW7RZe1a0xePBg+8cff1zi888++8z+2GOPFfvsjTfesE+dOtVut9vtL7/8sv21114r9v0TTzxhX7Nmjd1ut9u7du1qX7dunfHdqVOn7C1atLD/9ttvt/39j4uLs3fr1s3+5ZdfFisIbuU9u9q5K5Oy7l9ubq69devW9v/3//5fqb/T/bvk559/to8YMcJus9mMz7766iu7v7+//g6W05Xuof4e3hp65XebOnjwIB4eHtx9993GZ02aNOHYsWNkZWXdwiu7uQoLC/n+++9JTEzkf/7nf+jatSuvv/46586d4+DBgzRv3rzY8U2bNuXAgQMA/PTTT2V+n52dzfHjx4t9f+edd1K3bl3S0tJu+/vfuXNn/vOf//DXv/612Oe38p5d7dyVSVn378CBA9hsNhYtWkTHjh15/PHH+eCDDygsLAR0/4p4e3uzdOlSXFxcjM82bdpE69at9XewnK50D/X38NZQQXWbOn/+PCaTqdhnRX++cOHCrbikW+LMmTM88MADPP7448THx7N69Wp+/fVXpkyZUuo9cnNzM+7Plb4/f/48ADVr1izx/fnz52/7++/p6Um1atVKfH4r79nVzl2ZlHX/srOzad++PaGhoXzzzTeEh4ezcuVKPvzwQ0D3rzR2u50FCxawZcsWXn31Vf0dvAZ/vIf6e3hrlPxvBLkt1KxZk5ycnGKfFf25Vq1at+KSbok777yTVatWGX82mUxMmTKFp556igEDBnDx4sVix1+8eNG4PyaTqdTv77jjDuO/EP54j4t+b7fbq+T9N5lMZGdnF/vsZt2zq537dtCpUyc6depk/LlNmzYMGzaM+Ph4wsLCdP/+wGq1Mn36dL7//ns+/vhjWrRoob+DFVTaPWzRooX+Ht4CekJ1m2rWrBlnz54lIyPD+Oznn3/Gy8uL2rVr38Iru7kOHDhAREQEdrvd+CwvLw9nZ2fatGnDwYMHix3/008/0axZM+DSPSzr+7p163L33Xfz008/Gd+dOnWKs2fP0rx58yp7/5s3b37L7tnVzn07+Oqrr1i9enWxz/Ly8nBzcwN0/y7322+/MXDgQKxWK2vWrKFFixaA/g5WRFn3UH8Pb5Fbt31Lrtczzzxjf+mll+zZ2dlGp8WiRYtu9WXdVOnp6fa2bdvaP/jgA3t+fr796NGj9qeeeso+Y8YM+5kzZ+yBgYH2ZcuW2fPy8uw7duyw+/n52Xfs2GG32+1G98mOHTuMTpd27drZMzMz7Xa73b5gwQJ7r1697L/99pvR6TJkyBDj3FXl/l++qfpW3rOrnbuyuvz+bd682d6mTRv79u3b7YWFhfY9e/bYg4KC7F988YXdbtf9K3L27Fl7t27d7NOmTbMXFBQU+05/B8vnSvdQfw9vDRVUt7FTp07Zx40bZ2/fvr39oYcesr/zzjvFOj7+LMxms/3pp5+2+/n52R966CH7m2++ab948aLdbrfbU1NTje8effRRe2xsbLHffvHFF/bHH3/c3rZtW3tISIjdYrEY3+Xl5dnDw8PtXbp0sfv7+9tffPFFe0ZGhvF9Vbn/f2z7v5X37Grnroz+eP8+/fRT+2OPPWZ/8MEH7Y8++miJDlTdP7v9ww8/tDdv3tz+4IMP2tu2bVvsH7tdfwfL42r3UH8Pbz4nu/2ydyUiIiIiUmHaQyUiIiJynVRQiYiIiFwnFVQiIiIi10kFlYiIiMh1UkElIiIicp1UUImIiIhcJxVUIiIiItdJBZWIiIjIdVJBJSIiInKdVFCJiIiIXCcVVCIiIiLX6f8DdlWVwfZi1JIAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAiEAAAGvCAYAAACNRQWyAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAAApM0lEQVR4nO3de3yMd6LH8W8uHEkajXusKm0kemjJEBmXqixKpY07tUXbPWiPtNrYilK76tIoB1WXclwW63JKG5fqEZSuYitNahU9ljbJ1q3ZqiQidxF5zh9e5tVpggyRH/F5v15er85vnpnn55dp55NnnnnqZlmWJQAAgArmbnoCAADg3kSEAAAAI4gQAABgBBECAACMIEIAAIARRAgAADCCCAEAAEYQIQAAwAgiBADuMFxDEvcKIgSV1vz589W0adNyea5x48apc+fOLj+uadOmmj9/vuP20KFD1bRpU8efRx55RDabTX379tXq1at1+fLlcpnvzYqPj9err76qjh07qmXLlurevbumT5+utLQ0o/P69TpWVj/99JNefvll/fjjj6anAlQIT9MTAO41zZo109tvvy1Junz5si5cuKA9e/Zo2rRp+vvf/645c+bIzc2twuc1e/ZsLV26VE899ZQmTJggPz8/ff/991q6dKk+++wzrV69Wg0aNKjwed1L9u/fry+++EJ/+tOfTE8FqBBECFDB7rvvPgUHBzuNde7cWQ899JDeffddde7cWT179qzQOcXFxWnJkiUaP368XnzxRcd427ZtFRYWpt69e2vq1Kn67//+7wqdF4DKjY9jcM/YuHGjmjVrpsOHD+vZZ5/VY489prCwMC1dutRpuwsXLmj8+PGy2+1q06aNZs6cqeLi4ts+v6FDh6pu3bpat25dqfdfvHhRISEhmjZtmtN4cXGxHn/8cU2ePFmSdPToUb3wwgtq3bq1bDabXnzxRR0+fPi6+168eLGaNGmiF154ocR9Dz74oMaOHavWrVs71uHy5ctau3atIiIi1KJFC4WFhWnWrFm6ePGi43GlfYR15swZNW3aVBs3bnSMpaSkaMSIEWrVqpXat2+vOXPmaPz48Ro6dKjTY3NycjRhwgSFhobKZrPptddeU3p6+nX/Xr9WXFysuXPnqnPnznr00UfVuXNnvffee7p06ZIkqV+/fho0aFCJxw0bNswxn6FDh2rixIla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yf8iVFLe3t7Kz893Grt628fHx8SUjKhdu7bWrl3ruO3l5aXo6GgNHDhQffv2VUFBgdP2BQUFjvXx8vIq9f4aNWo4/iX69RpffbxlWZVy/b28vJSdne00VlFrdqN93y06dOigDh06OG63aNFCL7zwguLi4jR8+HDW8BdycnI0fvx4HT16VGvWrFHTpk15DbqotDVs2rQpr0FD7pkjIYGBgcrMzFRaWppjLCUlRf7+/vL19TU4s4p1/PhxzZo1S5ZlOcYKCwvl7u6uFi1aKCkpyWn75ORkBQYGSrqyhte6//7771e9evWUnJzsuO/cuXPKzMxUUFBQpV3/oKAgY2t2o33fLXbt2qV169Y5jRUWFqpatWqSWMOrTp06pX79+iknJ0exsbFq2rSpJF6DrrjWGvIaNMjc6SgV73e/+501evRoKzs723GG8rx580xPq0L961//soKDg60lS5ZYly5dsn788Udr4MCB1ltvvWVlZGRYISEh1ooVK6zCwkIrPj7estlsVnx8vGVZluOs7fj4eMcZ4m3atLHOnz9vWZZlzZkzx3rmmWesU6dOOc4QHzJkiGPflWX9f3lSpck1u9G+72S/XMPPPvvMatGihbV//36ruLjYOnjwoGW3263NmzdblsUaWtaVb1CEhYVZ48aNsy5fvux0H6/BsrneGvIaNOeeipBz585Zo0aNskJDQ622bdta06dPdzpT+l6RkJBgPfvss5bNZrPatm1rTZ061SooKLAsy7KOHDniuK9Lly7Whg0bnB67efNmq3v37lZwcLDVv39/69ChQ477CgsLrZkzZ1odO3a0WrVqZY0cOdJKS0tz3F9Z1v/XXy81uWY32ved6tdr+OGHH1rdunWzWrZsaXXp0qXEt7fu9TVcvny5FRQUZLVs2dIKDg52+mNZvAbL4kZryGvQDDfL+sVxeQAAgApyz5wTAgAA7ixECAAAMIIIAQAARhAhAADACCIEAAAYQYQAAAAjiBAAAGAEEQIAAIwgQgAAgBFECAAAMIIIAQAARvw/8RByMQ1yJqoAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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9WpNmDBB48aNO++8qKgo9e/fX76+voqNjVXnzp21cuXKi17sxeIzIAAAnN2Tn/zHynk9jpAePXpo/fr1io2NPee87OxshYWFuY21adNG+/bt8/SUAACgHvL42zHNmjW7oHklJSVyOBxuY40aNVJpaamnpwQAAPXQZbtPiMPhUHl5udtYeXm5/P39L9cpz6pn81o/JQAAdcZbg9taOe9li5CwsDBlZbl/jzw7O1uhoaGX65Rn9cbDNfv1IwAA6hNb9wu5bBESHx+vzMxMpaWlqbKyUmlpacrMzFRCQsLlOuU51fT3oAEAqA9s/n2s0QhxOp369NNPJUkhISH629/+poULF6pz586aP3++UlJS1Lp165o8pUe2j+/FWzMAcAZtvC587ht31f4V7bokyM9Lnz3eRe0a2l7Jb3dMDWzkc8Zjbw1ua/3/oF9Vd0wFAACXV63cMRUAAOBSECEAAMAKIgQAAFhBhAAAACuIEAAAYAURAgAArCBCAACAFUQIAACwgggBAABWECEAAMAKIgQAAFhBhAAAACuIEAAAYAURAgAArCBCAACAFUQIAACwgggBAABWECEAAMAKIgQAAFhBhAAAACuIEAAAYAURAgAArCBCAACAFUQIAACwgggBAABWECEAAMAKIgQAAFhBhAAAACuIEAAAYAURAgAArCBCAACAFUQIAACwgggBAABWECEAAMAKIgQAAFhBhAAAACuIEAAAYAURAgAArCBCAACAFUQIAACwgggBAABWECEAAMAKIgQAAFhBhAAAACuIEAAAYAURAgAArCBCAACAFUQIAACwgggBAABWECEAAMAKIgQAAFhBhAAAACuIEAAAYAURAgAArCBCAACAFUQIAACwgggBAABWeBwh+fn5SkxMVFRUlKKjozV16lRVVlaece7f//539e3bV506dVJcXJzWrVt3yQsGAAD1g8cRkpSUJD8/P23btk2pqalKT0/XkiVLqs3bsmWLFi5cqMWLF2vnzp16+umnlZSUpJ9++qkm1g0AAOo4jyLkwIEDyszM1MSJE+VwONSyZUslJiZq2bJl1eb+97//lTHG9ePj46MGDRrI19e3xhYPAADqLo+KICsrS4GBgWrevLlrLCQkRIcPH9bx48fVuHFj1/hdd92lVatWKTY2Vj4+PvLy8tLMmTMVHBxcc6sHAAB1lkdXQkpKSuRwONzGTj8uLS11Gz958qRuvfVWffTRR9q1a5eSk5M1adIkff/995e4ZAAAUB94FCF+fn4qKytzGzv92N/f3238tddeU2hoqDp06KCGDRvq3nvvVUREhFavXn2JSwYAAPWBRxESGhqqwsJC5eXlucZycnIUHBysgIAAt7mHDx9WRUWF25ivr68aNGhwCcsFAAD1hUcR0qpVK0VGRmratGkqLi7WoUOHNH/+fA0dOrTa3L59+2rp0qXau3evTp06pS+//FIZGRmKjY2tscUDAIC6y8sYYzx5Ql5enpKTk5WRkSFvb28NHjxYEyZMkI+Pj5xOp6ZMmaL4+HhVVlZqwYIFWr16tY4dO6abb75Z48aNU8+ePT1aYF5ekTxbIQAAsMXLSwoKCjj/RF1EhNQ2IgQAgLrDkwjhtu0AAMAKIgQAAFhBhAAAACuIEAAAYAURAgAArCBCAACAFUQIAACwgggBAABWECEAAMAKIgQAAFhBhAAAACuIEAAAYAURAgAArCBCAACAFUQIAACwgggBAABWECEAAMAKIgQAAFhBhAAAACuIEAAAYAURAgAArCBCAACAFUQIAACwgggBAABWECEAAMAKIgQAAFhBhAAAACuIEAAAYAURAgAArCBCAACAFUQIAACwgggBAABWECEAAMAKIgQAAFhBhAAAACuIEAAAYAURAgAArCBCAACAFUQIAACwgggBAABWECEAAMAKIgQAAFhBhAAAACuIEAAAYAURAgAArCBCAACAFUQIAACwgggBAABWECEAAMAKIgQAAFhBhAAAACuIEAAAYAURAgAArCBCAACAFUQIAACwgggBAABWECEAAMAKjyMkPz9fiYmJioqKUnR0tKZOnarKysozzs3MzNR9990np9Op3r17a+HChZe8YAAAUD94HCFJSUny8/PTtm3blJqaqvT0dC1ZsqTavJycHD3xxBN68MEHtXPnTi1cuFDvvPOOvvzyy5pYNwAAqOO8jDHmQicfOHBAAwcO1NatW9W8eXNJUlpammbOnKlNmza5zX3ttddUWFio2bNnu8Z+/PFHXXvttWrWrNkFLzAvr0gXvkIAAGCTl5cUFBRwQXM9uhKSlZWlwMBAV4BIUkhIiA4fPqzjx4+7zd2zZ49uuukmPfvss4qOjtagQYOUmZnpUYAAAID6y6MIKSkpkcPhcBs7/bi0tNRt/NixY3rvvfcU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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAiEAAAGvCAYAAACNRQWyAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAAAp5klEQVR4nO3ce1RVdf7/8RcXiQPiJS84qV9NBFumxlEUS00GNYsCzEFtasj5Tpdv4jjRKGVjadpX0qXlDDaa1pSZrLQomxopvzaTl5kxSE1tOWMD5K2YTFAUBMQD+/eHP8945HYOIB/E52Mt1vJ89mfv/T7vc+S82Gfv7WVZliUAAIBm5m26AAAAcG0ihAAAACMIIQAAwAhCCAAAMIIQAgAAjCCEAAAAIwghAADACEIIAAAwghACALVoqfdybKl1AZ4ihAANtHz5cvXr169JtjV79mxFR0d7vF6/fv20fPly5+PExET169fP+XPTTTfJbrdr4sSJeuutt1RZWdkk9TbE7t279dhjjykyMlIDBgxQVFSUnn76aR09etRYTbU5c+aMnnrqKe3atcs5lpiYqMTERINV1VwXcDUjhACtTP/+/bVhwwZt2LBB6enpevHFFzVw4EClpqZq5syZRv6K3rlzpx588EH5+fnpf//3f/WHP/xB06dP1969ezVp0qQWF0T++c9/6oMPPlBVVZXpUly01LqAhvI1XQCAptW2bVuFh4e7jEVHR+vGG2/UCy+8oOjoaMXFxTVrTa+88ooGDhyotLQ051hkZKRGjx6tcePG6Y033tC8efOatSYA5nEkBGgi77//vvr37699+/ZpypQpGjhwoKKiovTqq6+6zDt9+rSefvppRUZGaujQoVqyZEmz/GWbmJiorl27av369TUuP3funCIiIpSamuoyXlVVpZEjR2r+/PmSpAMHDmjq1KkaMmSI7Ha7fv7zn2vfvn117rugoKDG8a5du+qZZ57RiBEjnGOVlZVKT09XbGysBg0apKioKC1dulTnzp1zzpk9e7amTp2qefPmKSIiQvfee68cDoeqqqq0evVqjRs3TgMGDND48eP11ltvudWfi7KysvTggw9Kkh588EGXr2Asy9Krr76qqKgoDRo0SFOmTNFXX33lXL58+XKNGzdOL7/8siIjIzV27FidOnXKreckSe+++64mTpyo8PBwDRo0SPHx8crMzKy3LuBqRQgBmlBVVZWSk5MVExOj1atXa8iQIVq6dKl27NjhXP7www9r69atmjVrlhYvXqwvv/zS+UFzJfn4+OjWW2/V/v375XA4qi2/7rrrNH78eH388ccuoSgrK0snTpxQfHy8SkpK9PDDD6tjx45KS0vTsmXLVFZWpoceekjFxcW17jsqKkpffvmlEhMTlZGRoWPHjjmXTZo0SWPHjnU+njt3rlJTUxUdHa2VK1fqgQce0Lp165SUlOTyVdKuXbt05MgRLV++XNOnT5evr6+ee+45paWlKS4uTq+88oruvPNOpaam6ve//73bfbr55ps1d+5cZy2XHqHZvXu3tmzZomeffVaLFy/W8ePH9dhjj7n0Mz8/X1u2bNFLL72k5ORkdezY0a3nlJ6errlz52rMmDFatWqVlixZojZt2iglJUX5+fl11gVcrfg6BmhClmUpKSlJkyZNkiQNGTJEW7Zs0datWzVq1Cht375d+/fv16pVqxQVFSVJGj58eINOSm2Izp076/z58yoqKlLnzp2rLY+Pj1dGRoZ27dqlYcOGSZI++ugj9erVS+Hh4dq7d69OnjypxMREDRkyRJLUp08frV+/XiUlJQoKCqpxv48//riKi4v13nvvKTs7W5IUHBysqKgoTZ06VSEhIZKk3NxcZWRkKDk5WdOmTZMkjRgxQl27dtWTTz6p7du3a/To0ZIkh8Oh+fPnq1evXpKkQ4cO6Z133tGvf/1rPfroo5KkkSNHysvLS6tWrdL999+vjh071tujtm3bqm/fvpKkvn37Ov8tSX5+flq9erU6dOggSSopKdEzzzyj3Nxc3XTTTc66nnrqKd12220ePadjx47pF7/4haZPn+7cX48ePTRx4kTt2bNH99xzT611AVcrjoQATcxutzv/7efnp+uvv16lpaWSLvz13qZNG91+++3OOQEBAc4P1ubi5eVV4/jQoUPVvXt3bdq0SZJUUVGhLVu2OM8hCQ0N1fXXX69p06Zp3rx5+stf/qIuXbroySef1I9+9KNa9+fn56cFCxZo69atWrhwoWJjY2VZljZs2KD4+Hht3rxZkpwBJTY21mX9u+++Wz4+PsrKynKO+fv767/+67+cjz///HNZlqXo6Gg5HA7nT3R0tM6dO6fdu3c3oFOu+vbt6wwg0oWQIKnaUaCwsDDnv919TrNnz1ZKSoqKi4v11Vdf6aOPPlJ6erok6fz5842uHWiJOBICNDF/f3+Xx97e3s5D7qdPn1aHDh3k7e2a/7t06dIstR0/flz+/v4uH6SX8vLyUmxsrN555x09++yz2r59u86cOaP4+HhJUmBgoNLT07Vy5UplZmZq/fr1stlsiouL05w5c3TdddfVuf8uXbooISFBCQkJki581TNr1izNnz9f48aN0+nTp53zLuXr66uOHTu6fNh36tTJJUwVFRVJuvDhXttzb6yAgACXxxdfx8vP6bn0KJO7z+no0aOaO3euPv/8c/n6+qpPnz7OS8C5LwhaK0II0Iw6duzoPFHRx8fHOX7xA/RKqqysVHZ2tgYPHuyy78vFx8frlVdeUVZWlv70pz9p8ODB6tmzp3N5nz59tGTJElVWVmr//v364x//qLfffls9evRwfg1yqX379mnatGlasmSJywmo0oUrZB566CG98MILOnXqlNq3by9JOnHihPMog3ThSMCpU6fq/DqlXbt2kqQ333xTgYGB1ZbfcMMNta57JbnznKqqqvToo4+qTZs2euedd9S/f3/5+voqNzdXH374oZG6gebA1zFAM7r11lvlcDj06aefOscqKir0t7/97Yrve/369frhhx/005/+tM55ffr00cCBA7Vp0yZt3brVeRREkj755BMNHz5cJ06ckI+Pj+x2u5577jm1a9dO33//fY3b6927t8rKyrR27doarwI6dOiQunTpouuvv97lPJRLbdq0SZWVlc7zUGoydOhQSdKpU6c0cOBA509RUZF++9vfehT06gppnnLnOZ06dUqHDh1SQkKCBg0aJF/fC38fbt++XdJ/jrQ0ZV1AS8CREKAZ3XrrrRo5cqSeeeYZFRYWqnv37lq7dq1OnjypTp06Nck+SkpKtHfvXkkXPrxOnTqlv/71r9qwYYPi4uJ0xx131LuNCRMmKDU1Vd7e3rrrrruc44MHD1ZVVZWmT5+uRx99VIGBgfr4449VXFxc63bbt2+vp556SvPmzdP999+vyZMnq2fPniouLtaWLVu0ceNGLV26VF5eXurbt6/uvfdevfzyyyovL1dkZKT++c9/Oi95HTVqVK01h4WFKS4uTs8++6y+++47DRgwQIcOHdKyZcvUo0cP9e7d2+0eXjzBduvWrWrfvr3zpNOGcOc5eXt7q3v37kpPT1e3bt3Url07/fWvf9Wb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erRHBTblzcokLtcFAOByTXnHVE9uVnZN3TEVAABcWVf0jqkAAABNgRACAACMIIQAAAAjCCEAAMAIQggAADCCEAIAAIwghAAAACMIIQAAwAhCCAAAMMLXdAH18fIyXQEAAHCXJ5/bLf627QAAoHXi6xgAAGAEIQQAABhBCAEAAEYQQgAAgBGEEAAAYAQhBAAAGEEIAQAARhBCAACAEYQQAABgBCEEAAAYcU2FkMLCQiUlJSkiIkKRkZFauHChHA6H6bKaXWZmpvr37y+73e78SUlJkSTt27dPkyZNkt1uV3R0tN59912XdTdu3Khx48YpPDxcEydO1JdffulcVllZqcWLF+u2226T3W7XtGnT9MMPPziXt4b+nzx5UuPGjVNWVpZzzGTP6tt3S1RTD+fNm6cBAwa4vCc3bNjgXE4PpYMHD+q///u/NWzYMI0YMUJPPvmkTp48KYn3oLvq6iHvQUOsa8jPfvYza+bMmVZpaal19OhR6+6777ZeffVV02U1u0WLFlmzZ8+uNl5UVGQNGzbMWrdunXX+/Hnr73//u2W32619+/ZZlmVZn3/+uWW3261du3ZZFRUV1htvvGFFRkZapaWllmVZ1vLly63Y2FgrPz/fKi4utpKTk61HHnnEuf2rvf+7du2yxo4da4WFhVmff/65ZVlme1bfvluimnpoWZZ17733Wu+//36N69BDyyorK7NGjBhh/e53v7POnTtnnTx50nrkkUes//mf/+E96Ka6emhZvAdNuWZCyOHDh62wsDDr+++/d45t2rTJioqKMliVGQ888IC1bt26auPvvPOOdccdd7iMzZ0713ryyScty7KsmTNnWs8884zL8jvvvNPKyMiwLMuybr/9duvDDz90Ljtx4oTVr18/6+jRo1d9/99//30rKirK2rRpk8sHqMme1bfvlqa2Hp47d866+eabrX/96181rkcPLSsvL8966KGHLIfD4Rz79NNPrcGDB/MedFNdPeQ9aM4183VMTk6OOnTooODgYOdYSEiI8vPzdebMGYOVNa+qqiodOHBAW7du1Y9//GPdfvvtevbZZ3X69Gnl5OQoLCzMZX7fvn118OBBSVJubm6ty4uLi/X999+7LO/cubPat2+vr7/++qrv/8iRI7VlyxbFxMS4jJvsWX37bmlq6+HBgwflcDiUlpam2267TePHj9fq1atVVVUliR5KUp8+ffTaa6/Jx8fHObZ582bdfPPNvAfdVFcPeQ+ac82EkLNnz8pms7mMXXxcWlpqoiQjTp48qf79+2v8+PHKzMzU+vXrdfjwYaWkpNTYI39/f2d/6lp+9uxZSVJAQEC15WfPnr3q+9+lSxf5+vpWGzfZs/r23dLU1sPi4mINGzZMiYmJ2rZtm5YsWaK33npLr7/+uiR6eDnLsrRs2TJ99tlnmjNnDu/BBri8h7wHzan+G6GVCggIUFlZmcvYxceBgYEmSjKic+fOSk9Pdz622WxKSUnR5MmTNXHiRJWXl7vMLy8vd/bHZrPVuLxjx47O/0SX9/ji+pZltcr+22w2FRcXu4w1V8/q2/fVYsSIERoxYoTz8aBBgzR16lRlZmbq4YcfpoeXKCkp0dNPP60DBw5o3bp16tevH+9BD9XUw379+vEeNOSaORISGhqqoqIiFRQUOMfy8vLUrVs3BQUFGayseR08eFBLly6VZVnOsYqKCnl7e2vQoEHKyclxmZ+bm6vQ0FBJF3pY2/L27dsrODhYubm5zmUnTpxQUVGRwsLCWm3/w8LCjPWsvn1fLT799FOtX7/eZayiokL+/v6S6OFFR48e1U9+8hOVlJQoIyND/fr1k8R70BO19ZD3oEHmTkdpfj/96U+tJ554wiouLnaeoZyWlma6rGb173//2woPD7dWr15tnT9/3vruu++syZMnW7/5zW+skydPWhEREdYbb7xhVVRUWDt37rTsdru1c+dOy7Is51nbO3fudJ4hPnToUOvUqVOWZVnWsmXLrHvuucc6evSo8wzxn/3sZ859t5b+X3pSpcme1bfvluzSHv7f//2fNWjQIOvvf/+7VVVVZe3Zs8eKjIy0PvjgA8uy6KFlXbiCIioqypo9e7ZVWVnpsoz3oHvq6iHvQXOuqRBy4sQJa8aMGdawYcOs4cOHW4sWLXI5U/pakZWVZU2ZMsWy2+3W8OHDreeff94qLy+3LMuy9u/f71w2ZswY67333nNZ94MPPrDGjx9vhYeHWwkJCdbevXudyyoqKqwlS5ZYo0aNsgYPHmxNmzbNKigocC5vLf2//PJSkz2rb98t1eU9fPvtt6077rjDuuWWW6wxY8ZUu3rrWu/h66+/boWFhVm33HKLFR4e7vJjWbwH3VFfD3kPmuFlWZcclwcAAGgm18w5IQAAoGUhhAAAACMIIQAAwAhCCAAAMIIQAgAAjCCEAAAAIwghAADACEIIAAAwghACAACMIIQAAAAjCCEAAMCI/wdILNeBH6er9wAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAiEAAAGvCAYAAACNRQWyAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAAAviUlEQVR4nO3de1xVVf7/8TcXSUAUC9NGbSzj0JgXjqLkpTS0nCzRcbxU6uSUNklZeCFvaWnp6Gial1FLR8v0m6ZpzUyYv6xknDLQUptpHhZY3nIyUVFAEIH1+8Mv5+uR2zmILMXX8/HgUWfttdf6nHV28nbvfXY+xhgjAACAKuZruwAAAHBtIoQAAAArCCEAAMAKQggAALCCEAIAAKwghAAAACsIIQAAwApCCAAAsIIQAqDK8GzEqwOfE6oKIQRXnQULFigiIqJSxho3bpxiYmK83i8iIkILFixwvR48eLAiIiJcP7fffrucTqf69Omjt956SwUFBZVSb0V8+eWXevLJJxUdHa3mzZurS5cuGj9+vA4ePOjWLyYmRuPGjbusdfzhD3+4bONfSWbPnq3o6GhFRkbqvffe82ifyjyuL8XixYv1l7/8xfX6SqkL1ZO/7QKA6qJZs2Z64YUXJEkFBQU6deqUkpKSNH36dH355ZeaO3eufHx8qrSm7du3a+jQoeratatefvll1a5dWwcPHtTy5cvVr18/rVu3TjfffHOV1LJu3TqlpaVVyVw2fffdd1q6dKn69++vXr166dZbb7VdkldeffVVPf3007bLwDWCEAJUklq1aikyMtKtLSYmRrfccov++Mc/KiYmRrGxsVVa05IlS9SiRQvNnz/f1RYdHa3OnTvr3nvv1YoVK1zBCZUjIyNDkvTAAw8oKirKbjHAFY7LMbjqbdiwQc2aNdOePXs0YMAAtWjRQl26dNHSpUvd+p06dUrjx49XdHS02rZtq1mzZqmwsPCy1zd48GDdeOONWrNmTYnbz549q6ioKE2fPt2tvbCwUJ06ddKUKVMkSd98840effRRtWnTRk6nU0OGDNGePXvKnDs9Pb3E9htvvFHPP/+8Onbs6NZ+7tw5/elPf1LHjh0VGRmpxx57TAcOHHDr89lnn+mRRx5RmzZtFB0drdGjR+u///2va3vR57Fu3Tp16tRJd999t5555hlt3LhRP/74oyIiIrRhwwYdPnxYERER2rRpk5555hk5nU61bdtWEydOVHZ2ttuc69at0wMPPOC6nLRgwQLl5+e7tp84cUJjxoxRx44d1aJFC/Xq1cvtMkhhYaHmzZunmJgYNW/eXDExMZozZ47OnTtX5vqVJDExUX369JHT6VTHjh01efJknTp1StL5SxeDBw+WJD366KMVutS3ZcsWde/eXS1atFC/fv20fft217bk5GRFRERozZo1uueee9ShQwf985//lCTt3LlTgwYNUqtWrdSuXTuNHTtWJ06ccBt7x44devzxx9W2bVvXOixYsMD130HRZZeFCxcWuwSzdetWxcbGqkWLFurevbvHl5mAshBCUC0UFhYqPj5ePXr00Ouvv642bdpo9uzZ2rZtm2v70KFDtXXrVo0ZM0YzZ87Url27lJiYeNlr8/PzU/v27fX111+7/eIsct1116l79+7atGmTWyhKTk7WsWPH1KtXL2VlZWno0KGqW7eu5s+fr7lz5yonJ0ePP/64MjMzS527S5cu2rVrlwYPHqz169fr0KFDrm39+vVTt27d3PonJiYqNTVVM2bM0OTJk/Wvf/1LI0eOdG1///339dhjj6l+/fqaM2eOxo8fr127dmnAgAE6fvy4q19BQYGWLFmil19+WfHx8RozZow6d+6sevXqae3aterSpYur7wsvvKCGDRtq0aJFGjp0qN59910tWbLEtf21117TpEmT1L59ey1ZskQDBw7U0qVLNXnyZFefhIQEpaWlacqUKXr99dfVrFkzjR07VsnJyZKkpUuXavXq1Xrqqae0fPlyPfzww1q2bJnbPJ5YtGiRRo4cqVatWmn+/Pl66qmntHnzZg0ePFi5ubnq16+fq67Jkydr4cKFXo0vSRMmTNDvfvc7LViwQMHBwRo2bFixy1hz587V2LFjNXbsWEVGRmrHjh0aMmSIatasqVdffVUTJkxQSkqKfve73yk3N1eStHfvXg0ZMkShoaGaO3euFi9erNatW2vhwoX64IMPJElr166VJPXt29f170UmT56sIUOGaPHixbrxxhs1btw47d271+v3B7gxwFVm/vz5xuFwuF6/++67xuFwmHfeecfVdvbsWdOiRQszdepUY4wxn376qXE4HObTTz919cnOzjbR0dHmnnvu8boGh8Nh5s+f73o9aNAgM2jQoFL7z5w50zgcDnPs2LEStycnJxuHw2GS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AAAAKwghAADACkIIAACwghACAACsIIQAAAArCCEAAMAKQggAALCCEAIAAKwghAAAACsIIQAAwApCCAAAsIIQAgAArCCEAAAAKwghAADACkIIAACwghACAACsIIQAAAArCCEAAMAKQggAALCCEAIAAKwghAAAACsIIQAAwApCCAAAsIIQAgAArCCEAAAAKwghAADACkIIAACwghACAACsIIQAAAArCCEAAMAKQggAALCCEAIAAKwghAAAACsIIQAAwApCCAAAsIIQAgAArCCEAAAAK7wOIcePH1dcXJyioqIUHR2tadOmKT8/v8S+KSkp6tevn5xOpzp37qzXXnvtkgsGAADVg9chJD4+XkFBQdq2bZvWr1+v7du364033ijWb9++fXriiSf0yCOP6KuvvtJrr72m5cuX68MPP6yMugEAwFXOxxhjPO184MAB3XffffrHP/6h+vXrS5ISExM1a9Ysffrpp259X3rpJWVkZOiVV15xtf3www+qVauW6tWr53GB6emZ8rxCAABgk4+PFBYW4lFfr86EpKamKjQ01BVAJKlp06Y6cuSITp8+7db366+/VqNGjTRq1ChFR0fr/vvvV0pKilcBBAAAVF9ehZDs7GwFBga6tRW9PnPmjFv7qVOntHLlSsXGxuqzzz7T1KlTNXPmTC7HAAAASV6GkKCgIOXk5Li1Fb0ODg52aw8ICFDXrl3VpUsX+fv7q23bturVq5c2bdp0iSUDAIDqwKsQEh4eroyMDKWnp7va9u3bpwYNGigkxP36T9OmTZWXl+fWVlBQIC9uQQEAANWYVyGkSZMmatOmjaZPn66srCwdOnRIixYtUt++fYv1feihh/Txxx/r/ffflzFGO3bs0N/+9jf16tWr0ooHAABXL6++HSNJ6enpmjp1qpKTk+Xr66vevXtrzJgx8vPzk9Pp1JQpUxQbGytJSkpK0vz58/XDDz/o+uuv19ChQ/XQQw95VSDfjgEA4OrhzbdjvA4hVY0QAgDA1eOyfUUXAACgshBCAACAFYQQAABgBSEEAABYQQgBAABWEEIAAIAVhBAAAGAFIQQAAFhBCAEAAFYQQgAAgBWEEAAAYAUhBAAAWEEIAQAAVhBCAACAFYQQAABgBSEEAABYQQgBAABWEEIAAIAVhBAAAGAFIQQAAFhBCAEAAFYQQgAAgBWEEAAAYAUhBAAAWEEIAQAAVhBCAACAFYQQAABgBSEEAABYQQgBAABWEEIAAIAVhBAAAGAFIQQAAFhBCAEAAFYQQgAAgBWEEAAAYAUhBAAAWEEIAQAAVhBCAACAFYQQAABgBSEEAABYQQgBAABWEEIAAIAVhBAAAGAFIQQAAFhBCAEAAFYQQgAAgBWEEAAAYAUhBAAAWEEIAQAAVhBCAACAFYQQAABgBSEEAABYQQgBAABWEEIAAIAVhBAAAGAFIQQAAFhBCAEAAFZ4HUKOHz+uuLg4RUVFKTo6WtOmTVN+fn6Z+3z33Xdq1aqVkpOTK1woAACoXrwOIfHx8QoKCtK2bdu0fv16bd++XW+88Uap/XNycjR69Gjl5uZeSp0AAKCa8SqEHDhwQCkpKUpISFBgYKAaN26suLg4rV69utR9pkyZom7dul1yoQAAoHrxKoSkpqYqNDRU9evXd7U1bdpUR44c0enTp4v1f++993TgwAE9/fTTl14pAACoVvy96Zydna3AwEC3tqLXZ86cUe3atV3t+/bt09y5c/X222/Lz8+vEkoFAADViVdnQoKCgpSTk+PWVvQ6ODjY1Xb27FmNHDlSEyZM0C9+8YtKKBMAAFQ3XoWQ8PBwZWRkKD093dW2b98+NWjQQCEhIa62f/3rX9q/f78mTpyoqKgoRUVFSZKefPJJvfjii5VTOQAAuKr5GGOMNzs88sgjatCggaZOnaqTJ09q+PDh6t69u0aMGFHmfhEREVq5cqWio6O9KjA9PVPeVQgAAGzx8ZHCwkLK76gKfEV3/vz5ys/PV9euXdW/f3/dddddiouLkyQ5nU799a9/9XZIAABwDfL6TEhV40wIAABXj8t6JgQAAKAyEEIAAIAVhBAAAGAFIQQAAFhBCAEAAFYQQgAAgBWEEAAAYAUhBAAAWEEIAQAAVhBCAACAFYQQAABgBSEEAABYQQgBAABWEEIAAIAVhBAAAGAFIQQAAFhBCAEAAFYQQgAAgBWEEAAAYAUhBAAAWEEIAQAAVhBCAACAFYQQAABgBSEEAABYQQgBAABWEEIAAIAVhBAAAGAFIQQAAFhBCAEAAFYQQgAAgBWEEAAAYAUhBAAAWEEIAQAAVhBCAACAFYQQAABgBSEEAABYQQgBAABWEEIAAIAVhBAAAGAFIQQAAFhBCAEAAFYQQgAAgBWEEAAAYAUhBAAAWEEIAQAAVhBCAACAFYQQAABgBSEEAABYQQgBAABWEEIAAIAVhBAAAGAFIQQAAFhBCAEAAFYQQgAAgBWEEAAAYAUhBAAAWEEIAQAAVhBCAACAFV6HkOPHjysuLk5RUVGKjo7WtGnTlJ+fX2Lft99+W927d5fT6VT37t21evXqSy4YAABUD16HkPj4eAUFBWnbtm1av369tm/frjfeeKNYvy1btmjOnDmaOXOmvvrqK82YMUOvvvqqNm/eXBl1AwCAq5xXIeTAgQNKSUlRQkKCAgMD1bhxY8XFxZV4huPo0aMaNmyYIiMj5ePjI6fTqejoaO3YsaPSigcAAFcvf286p6amKjQ0VPXr13e1NW3aVEeOHNHp06dVu3ZtV/vAgQPd9j1+/Lh27Nih8ePHX2LJAACgOvDqTEh2drYCAwPd2openzlzptT9jh07pmHDhql58+Z68MEHK1AmAACobrwKIUFBQcrJyXFrK3odHBxc4j67d+9W3759dcstt2jx4sXy9/fq5AsAAKimvAoh4eHhysjIUHp6uqtt3759atCggUJCQor1X79+vYYMGaJHH31Ur7zyigICAi69YgAAUC14FUKaNGmiNm3aaPr06crKytKhQ4e0aNEi9e3bt1jfzZs368UXX9SCBQv02GOPVVrBAACgevAxxhhvdkhPT9fUqVOVnJwsX19f9e7dW2PGjJGfn5+cTqemTJmi2NhY9ezZU2lpaapZs6bb/j179tTUqVO9mC9T3lUIAABs8fGRwsKKXx0psa+3IaSqEUIAALh6eBNCeGw7AACwghACAACsIIQAAAArCCEAAMAKQggAALCCEAIAAKwghAAAACsIIQAAwApCCAAAsIIQAgAArCCEAAAAKwghAADACkIIAACwghACAACsIIQAAAArCCEAAMAKQggAALCCEAIAAKwghAAAACsIIQAAwApCCAAAsIIQAgAArCCEAAAAKwghAADACkIIAACwghACAACsIIQAAAArCCEAAMAKQggAALCCEAIAAKwghAAAACsIIQAAwApCCAAAsIIQAgAArCCEAAAAKwghAADACkIIAACwghACAACsIIQAAAArCCEAAMAKQggAALCCEAIAAKwghAAAACsIIQAAwApCCAAAsIIQAgAArCCEAAAAKwghAADACkIIAACwghACAACsIIQAAAArCCEAAMAKQggAALCCEAIAAKwghAAAACv8bRdQ1dq+8g/bJQAAcEWZ3LWxekbeUuXz+hhjTJXP6oX09ExVVoUEEAAASrdj9N2XPIaPjxQWFuJRX68vxxw/flxxcXGKiopSdHS0pk2bpvz8/BL7JiUlqWfPnoqMjNT999+vTz/91NvpKg0BBACAslX170qvQ0h8fLyCgoK0bds2rV+/Xtu3b9cbb7xRrN/+/fs1YsQIPfvss9q5c6dGjBih+Ph4HT16tDLq9goBBAAAz/xt9w9VNpdXIeTAgQNKSUlRQkKCAgMD1bhxY8XFxWn16tXF+m7cuFFRUVHq1q2b/P391aNHD7Vt21Zr166ttOIBAEDlmvrxoSqby6sQkpqaqtDQUNWvX9/V1rRpUx05ckSnT59265uWliaHw+HWdtttt2nv3r2XUC4AAKguvAoh2dnZCgwMdGsren3mzJly+9asWbNYPwAAcG3yKoQEBQUpJyfHra3odXBwsFt7YGCgcnNz3dpyc3OL9QMAAFeOyV0bV9lcXoWQ8PBwZWRkKD093dW2b98+NWjQQCEh7l/HcTgcSk1NdWtLS0tTeHj4JZRbMZXxlSMAAK4FVfm8EK9CSJMmTdSmTRtNnz5dWVlZOnTokBYtWqS+ffsW6xsbG6uUlBQlJiYqPz9fiYmJSklJUa9evSqteG8QRAAAKFtV/670+mFl6enpmjp1qpKTk+Xr66vevXtrzJgx8vPzk9Pp1JQpUxQbGytJ2rZtm2bPnq2DBw+qYcOGSkhIUOfOnb0qsDIfVibxdV0AAC5WmU9M9eZhZdfUE1MBAMDldVmfmAoAAFAZCCEAAMAKQggAALCCEAIAAKwghAAAACsIIQAAwApCCAAAsIIQAgAArCCEAAAAK/xtF1AeHx/bFQAAAE9583v7in9sOwAAqJ64HAMAAKwghAAAACsIIQAAwApCCAAAsIIQAgAArCCEAAAAKwghAADACkIIAACwghACAACsIIQAAAArrqkQcvz4ccXFxSkqKkrR0dGaNm2a8vPzbZdV5RITE9WsWTM5nU7XT0JCgiRpz5496tevn5xOp2JiYrRu3Tq3fTdu3Kh7771XkZGR6tOnj3bt2uXaVlBQoJkzZ6pDhw5yOp0aPny4fv75Z9f26rD+J06c0L333qvk5GRXm801K2/uK1FJa/jCCy+oefPmbsfk2rVrXdtZQ2nv3r36/e9/r3bt2qljx4567rnndOLECUkcg54qaw05Bi0x15BBgwaZ0aNHmzNnzpiDBw+aBx54wCxdutR2WVVuxowZZty4ccXaMzIyTLt27cyqVavMuXPnzOeff26cTqfZs2ePMcaYL774wjidTrNz506Tl5dnVqxYYaKjo82ZM2eMMcYsWLDA9OzZ0xw5csRkZmaa+Ph4M2zYMNf4V/v679y503Tr1s04HA7zxRdfGGPsrll5c1+JSlpDY4z5zW9+YzZs2FDiPqyhMTk5OaZjx45m3rx55uzZs+bEiRNm2LBh5g9/+APHoIfKWkNjOAZtuWZCyP79+43D4TA//fSTq+2DDz4wXbp0sViVHQMHDjSrVq0q1v7OO++Y++67z61t8uTJ5rnnnjPGGDN69Gjz/PPPu23/9a9/bdavX2+MMebuu+82f/3rX13bjh07ZiIiIszBgwev+vXfsGGD6dKli/nggw/cfoHaXLPy5r7SlLaGZ8+eNXfccYf57rvvStyPNTRm37595vHHHzf5+fmuti1btpjWrVtzDHqorDXkGLTnmrkck5qaqtDQUNWvX9/V1rRpUx05ckSnT5+2WFnVKiws1DfffKOtW7fqnnvu0d13361Jkybp1KlTSk1NlcPhcOt/2223ae/evZKktLS0UrdnZmbqp59+ctseFhamOnXq6Ntvv73q179Tp0766KOP1KNHD7d2m2tW3txXmtLWcO/evcrPz9f8+fPVoUMHde/eXa+//roKCwslsYaSdOutt2rZsmXy8/NztW3evFl33HEHx6CHylpDjkF7rpkQkp2drcDAQLe2otdnzpyxUZIVJ06cULNmzdS9e3clJiZqzZo12r9/vxISEkpco5o1a7rWp6zt2dnZkqSgoKBi27Ozs6/69a9Xr578/f2Ltdtcs/LmvtKUtoaZmZlq166dBg8erKSkJM2aNUtvvfWWli9fLok1vJgxRnPnztWnn36qiRMncgxWwMVryDFoT/E/EaqpoKAg5eTkuLUVvQ4ODrZRkhVhYWFavXq163VgYKASEhLUv39/9enTR7m5uW79c3NzXesTGBhY4va6deu6/iO6eI2L9jfGVMv1DwwMVGZmpltbVa1ZeXNfLTp27KiOHTu6Xrds2VKPPvqoEhMTNXToUNbwAllZWRo/fry++eYbrVq1ShERERyDXippDSMiIjgGLblmzoSEh4crIyND6enprrZ9+/apQYMGCgkJsVhZ1dq7d69mz54tY4yrLS8vT76+vmrZsqVSU1Pd+qelpSk8PFzS+TUsbXudOnVUv359paWlubYdO3ZMGRkZcjgc1Xb9HQ6HtTUrb+6rxZYtW7RmzRq3try8PNWsWVMSa1jk4MGD+u1vf6usrCytX79eERERkjgGvVHaGnIMWmTvdpSq9/DDD5uRI0eazMxM1x3K8+fPt11Wlfrvf/9rIiMjzeuvv27OnTtnfvzxR9O/f38zYcIEc+LECRMVFWVWrFhh8vLyzPbt243T6TTbt283xhjXXdvbt2933SHetm1bc/LkSWOMMXPnzjUPPvigOXjwoOsO8UGDBrnmri7rf+FNlTbXrLy5r2QXruH/+3//z7Rs2dJ8/vnnprCw0Hz11VcmOjravPfee8YY1tCY89+g6NKlixk3bpwpKChw28Yx6Jmy1pBj0J5rKoQcO3bMjBgxwrRr187ceeedZsaMGW53Sl8rkpOTzYABA4zT6TR33nmneemll0xubq4xxpivv/7ata1r167m3Xffddv3vffeM927dzeRkZGmb9++Zvfu3a5teXl5ZtasWeauu+4yrVu3NsOHDzfp6emu7dVl/S/+eqnNNStv7ivVxWv49ttvm/vuu8+0atXKdO3atdi3t671NVy+fLlxOBymVatWJjIy0u3HGI5BT5S3hhyDdvgYc8F5eQAAgCpyzdwTAgAAriyEEAAAYAUhBAAAWEEIAQAAVhBCAACAFYQQAABgBSEEAABYQQgBAABWEEIAAIAVhBAAAGAFIQQAAFjx/wG/fx9oBXm3tgAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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LfWNqQAA4Oa6qd+YCgAAUB+IEAAAYAQRAgAAjCBCAACAEUQIAAAwgggBAABGECEAAMAIIgQAABhBhAAAACM8TS+gNm5uplcAAADqypXf243+a9sBAEDzxMsxAADACCIEAAAYQYQAAAAjiBAAAGAEEQIAAIwgQgAAgBFECAAAMIIIAQAARhAhAADACCIEAAAYcUtFSE5OjqKjoxUeHq6IiAgtWbJEZWVlppfV4JKTk9W9e3fZ7XbHn9jYWEnS4cOH9fDDD8tutysyMlLvvvuu0223bdum+++/X2FhYRo/frwOHjzouK68vFzLly/XPffcI7vdrlmzZuns2bOO65vD/ufm5ur+++9XamqqY8zkntV27Maouj38/e9/r7vvvtvpnHz77bcd17OH0rFjx/Too4+qf//+uvfee/Xcc88pNzdXEudgXV1rDzkHDbFuIVOmTLHmzp1rFRcXWydPnrQefPBBa/369aaX1eCWLVtmzZ8/v8p4Xl6e1b9/f2vz5s3W5cuXrT179lh2u906fPiwZVmW9fXXX1t2u93at2+fVVpaar355ptWRESEVVxcbFmWZa1evdoaM2aMlZWVZRUUFFgxMTHWzJkzHfff1Pd/37591ogRI6yQkBDr66+/tizL7J7VduzGqLo9tCzL+sUvfmG999571d6GPbSskpIS695777VeffVV69KlS1Zubq41c+ZM68knn+QcrKNr7aFlcQ6acstEyPHjx62QkBDrhx9+cIxt377dGjp0qMFVmTF58mRr8+bNVcbfeecda+TIkU5jCxcutJ577jnLsixr7ty51gsvvOB0/c9+9jMrKSnJsizLGjx4sPXhhx86rjt37pwVGhpqnTx5ssnv/3vvvWcNHTrU2r59u9MvUJN7VtuxG5ua9vDSpUtWjx49rH//+9/V3o49tKzMzEzr8ccft8rKyhxjn3/+udWnTx/OwTq61h5yDppzy7wck56eLn9/fwUGBjrGunbtqqysLOXn5xtcWcOqqKjQ0aNHtWPHDg0bNkyDBw/Wiy++qAsXLig9PV0hISFO8++66y4dO3ZMkpSRkVHj9QUFBfrhhx+crg8ICFCbNm30zTffNPn9HzRokD777DONHj3aadzkntV27Mampj08duyYysrKtGrVKt1zzz0aNWqUXn/9dVVUVEhiDyWpS5cu2rBhgzw8PBxjn376qXr06ME5WEfX2kPOQXNumQgpKiqSzWZzGqu8XFxcbGJJRuTm5qp79+4aNWqUkpOTtXXrVh0/flyxsbHV7pG3t7djf651fVFRkSTJx8enyvVFRUVNfv/bt28vT0/PKuMm96y2Yzc2Ne1hQUGB+vfvr6lTp2rnzp2Kj4/XW2+9pTfeeEMSe3g1y7K0cuVKffnll1qwYAHn4HW4eg85B82p+hOhmfLx8VFJSYnTWOVlX19fE0syIiAgQImJiY7LNptNsbGxmjhxosaPH6+LFy86zb948aJjf2w2W7XXt23b1vE/0dV7XHl7y7Ka5f7bbDYVFBQ4jTXUntV27Kbi3nvv1b333uu43KtXL02bNk3JycmaMWMGe3iFwsJCPf/88zp69Kg2b96s0NBQzkEXVbeHoaGhnIOG3DLPhAQHBysvL0/Z2dmOsczMTAUFBcnPz8/gyhrWsWPHlJCQIMuyHGOlpaVyd3dXr169lJ6e7jQ/IyNDwcHBkn7cw5qub9OmjQIDA5WRkeG47ty5c8rLy1NISEiz3f+QkBBje1bbsZuKzz//XFu3bnUaKy0tlbe3tyT2sNLJkyf10EMPqbCwUElJSQoNDZXEOeiKmvaQc9Agc29HaXiPPPKINWfOHKugoMDxDuVVq1aZXlaD+v77762wsDDr9ddfty5fvmx999131sSJE63f/e53Vm5urhUeHm69+eabVmlpqZWSkmLZ7XYrJSXFsizL8a7tlJQUxzvE+/XrZ50/f96yLMtauXKl9fOf/9w6efKk4x3iU6ZMcRy7uez/lW+qNLlntR27MbtyD//3f//X6tWrl7Vnzx6roqLCOnDggBUREWG9//77lmWxh5b14ycohg4das2fP98qLy93uo5zsG6utYecg+bcUhFy7tw5a/bs2Vb//v2tAQMGWMuWLXN6p/StIjU11Zo0aZJlt9utAQMGWC+99JJ18eJFy7Is68iRI47rhg8fbv3tb39zuu37779vjRo1ygoLC7MmTJhgHTp0yHFdaWmpFR8fb913331Wnz59rFmzZlnZ2dmO65vL/l/98VKTe1bbsRurq/dwy5Yt1siRI63evXtbw4cPr/LprVt9D9944w0rJCTE6t27txUWFub0x7I4B+uitj3kHDTDzbKueF4eAACggdwy7wkBAACNCxECAACMIEIAAIARRAgAADCCCAEAAEYQIQAAwAgiBAAAGEGEAAAAI4gQAABgBBECAACMIEIAAIAR/z/GLtKx7S0x2AAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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1iKEAACAtQghAABgLUIIAABYixACAADWIoQAAIC1CCEAAGAtQggAAFiLEAIAANYihAAAgLUIIQAAYC1CCAAAWIsQAgAA1iKEAACAtQghAABgLUIIAABYixACAADWIoQAAIC1CCEAAGAtQggAAFiLEAIAANYihAAAgLUIIQAAYC1CCAAAWIsQAgAA1iKEAACAtQghAABgLUIIAABYixACAADWIoQAAIC1CCEAAGAtQggAAFiLEAIAANYihAAAgLUIIQAAYC1CCAAAWIsQAgAA1iKEAACAtQghAABgLUIIAABYixACAADWIoQAAIC1CCEAAGAtQggAAFiLEAIAANYihAAAgLUIIQAAYC1CCAAAWIsQAgAA1iKEAACAtQghAABgLUIIAABYixACAADWIoQAAIC1CCEAAGAtQggAAFiLEAIAANYihAAAgLUIIQAAYC1CCAAAWIsQAgAA1iKEAACAtQghAABgLUIIAABYixACAADWIoQAAIC1CCEAAGAtQggAAFiLEAIAANYihAAAgLUIIQAAYC1CCAAAWIsQAgAA1iKEAACAtQghAABgLUIIAABYixACAADWIoQAAIC1CCEAAGAtQggAAFiLEAIAANYihAAAgLUIIQAAYC1CCAAAWIsQAgAA1iKEAACAtQghAABgLUIIAABYixACAADWIoQAAIC1CCEAAGAtQggAAFiLEAIAANYihAAAgLUIIQAAYC1CCAAAWIsQAgAA1iKEAACAtQghAABgLUIIAABYixACAADWIoQAAIC1CCEAAGAtQggAAFiLEAIAANYihAAAgLUIIQAAYC1CCAAAWIsQAgAA1iKEAACAtQghAABgLUIIAABYixACAADWIoQAAIC1CCEAAGAtQggAAFiLEAIAANYihAAAgLUIIQAAYC1CCAAAWIsQAgAA1iKEAACAtQghAABgLUIIAABYixACAADWIoQAAIC1CCEAAGAtQggAAFiLEAIAANYihAAAgLUIIQAAYC1CCAAAWIsQAgAA1iKEAACAtQghAABgLUIIAABYixACAADWIoQAAIC1CCEAAGAtQggAAFiLEAIAANYihAAAgLUIIQAAYC1CCAAAWIsQAgAA1iKEAACAtQghAABgLUIIAABYixACAADWIoQAAIC1CCEAAGAtQggAAFiLEAIAANYihAAAgLUIIQAAYC1CCAAAWIsQAgAA1iKEAACAtQghAABgLUIIAABYixACAADWIoQAAIC1CCEAAGAtQggAAFiLEAIAANYihAAAgLUIIQAAYC1CCAAAWIsQAgAA1iKEAACAtQghAABgLUIIAABYixACAADWIoQAAIC1CCEAAGAtQggAAFiLEAIAANYihAAAgLUIIQAAYC1CCAAAWIsQAgAA1iKEAACAtQghAABgLUIIAABYixACAADWIoQAAIC1CCEAAGAtQggAAFiLEAIAANYihAAAgLUIIQAAYC1CCAAAWIsQAgAA1iKEAACAtQghAABgLUIIAABYixACAADWIoQAAIC1CCEAAGAtQggAAFiLEAIAANYihAAAgLUIIQAAYC1CCAAAWIsQAgAA1iKEAACAtQghAABgLUIIAABYixACAADWIoQAAIC1CCEAAGAtQggAAFiLEAIAANYihAAAgLUIIQAAYC1CCAAAWIsQAgAA1iKEAACAtQghAABgLUIIAABYixACAADWIoQAAIC1CCEAAGAtQggAAFiLEAIAANYihAAAgLUIIQAAYC1CCAAAWIsQAgAA1iKEAACAtQghAABgLUIIAABYixACAADWIoQAAIC1CCEAAGAtQggAAFiLEAIAANYihAAAgLUIIQAAYC1CCAAAWIsQAgAA1iKEAACAtQghAABgLUIIAABYixACAADWKl7UA7BR4/e3FvUQAAC44+wcElLox7xr3hHKzMzU8ePHi3oYfxoRBABA3oriNfKuCaFnn31WiYmJkqTk5GT5+PgU8YhuHREEAMCNFfZr5V0TQmfPnnX87O/vr127dhXhaG4dEQQAQP4U5mvmLYXQ4cOH5eXlpUWLFiksLEx+fn564YUXHJesEhMTFRERIX9/f7Vr107Lly93PDYnJ0cffPCBgoOD9cQTT2jMmDHq0aOHlixZIkn68ccf9corryg0NFQNGzZU27ZttXnzZknSiy++qKNHj2rMmDF65513lJSUJC8vL0lSZGSkhgwZ4jTOwYMHKyoqSpJ06NAhvfrqqwoMDFTz5s01ZcoUZWVl3eZ0AQCAe8ltvSOUkJCgZcuWad26dTp9+rQ++ugjpaSkqF+/fnr55ZeVlJSksWPHavz48dq2bZskac6cOVq+fLk+/fRTJSQkqFy5ck7v6gwcOFB169bVhg0blJycrKZNm+rtt9+WJMXFxenBBx9UVFSURo8e7TSWbt26aePG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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "ename": "TypeError",
     "evalue": "'value' must be an instance of str or bytes, not a float",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[124], line 6\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m j, col2 \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28menumerate\u001b[39m(df\u001b[38;5;241m.\u001b[39mcolumns):\n\u001b[0;32m      5\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m i \u001b[38;5;241m<\u001b[39m j:\n\u001b[1;32m----> 6\u001b[0m         plt\u001b[38;5;241m.\u001b[39mscatter(df[col1], df[col2])\n\u001b[0;32m      7\u001b[0m         plt\u001b[38;5;241m.\u001b[39mtitle(col1 \u001b[38;5;241m+\u001b[39m \u001b[38;5;124m'\u001b[39m\u001b[38;5;124m vs \u001b[39m\u001b[38;5;124m'\u001b[39m \u001b[38;5;241m+\u001b[39m col2)\n\u001b[0;32m      8\u001b[0m         plt\u001b[38;5;241m.\u001b[39mshow()\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\matplotlib\\pyplot.py:2862\u001b[0m, in \u001b[0;36mscatter\u001b[1;34m(x, y, s, c, marker, cmap, norm, vmin, vmax, alpha, linewidths, edgecolors, plotnonfinite, data, **kwargs)\u001b[0m\n\u001b[0;32m   2857\u001b[0m \u001b[38;5;129m@_copy_docstring_and_deprecators\u001b[39m(Axes\u001b[38;5;241m.\u001b[39mscatter)\n\u001b[0;32m   2858\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mscatter\u001b[39m(\n\u001b[0;32m   2859\u001b[0m         x, y, s\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mNone\u001b[39;00m, c\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mNone\u001b[39;00m, marker\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mNone\u001b[39;00m, cmap\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mNone\u001b[39;00m, norm\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mNone\u001b[39;00m,\n\u001b[0;32m   2860\u001b[0m         vmin\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mNone\u001b[39;00m, vmax\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mNone\u001b[39;00m, alpha\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mNone\u001b[39;00m, linewidths\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mNone\u001b[39;00m, \u001b[38;5;241m*\u001b[39m,\n\u001b[0;32m   2861\u001b[0m         edgecolors\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mNone\u001b[39;00m, plotnonfinite\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mFalse\u001b[39;00m, data\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mNone\u001b[39;00m, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs):\n\u001b[1;32m-> 2862\u001b[0m     __ret \u001b[38;5;241m=\u001b[39m gca()\u001b[38;5;241m.\u001b[39mscatter(\n\u001b[0;32m   2863\u001b[0m         x, y, s\u001b[38;5;241m=\u001b[39ms, c\u001b[38;5;241m=\u001b[39mc, marker\u001b[38;5;241m=\u001b[39mmarker, cmap\u001b[38;5;241m=\u001b[39mcmap, norm\u001b[38;5;241m=\u001b[39mnorm,\n\u001b[0;32m   2864\u001b[0m         vmin\u001b[38;5;241m=\u001b[39mvmin, vmax\u001b[38;5;241m=\u001b[39mvmax, alpha\u001b[38;5;241m=\u001b[39malpha, linewidths\u001b[38;5;241m=\u001b[39mlinewidths,\n\u001b[0;32m   2865\u001b[0m         edgecolors\u001b[38;5;241m=\u001b[39medgecolors, plotnonfinite\u001b[38;5;241m=\u001b[39mplotnonfinite,\n\u001b[0;32m   2866\u001b[0m         \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39m({\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mdata\u001b[39m\u001b[38;5;124m\"\u001b[39m: data} \u001b[38;5;28;01mif\u001b[39;00m data \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m \u001b[38;5;28;01melse\u001b[39;00m {}), \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n\u001b[0;32m   2867\u001b[0m     sci(__ret)\n\u001b[0;32m   2868\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m __ret\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\matplotlib\\__init__.py:1446\u001b[0m, in \u001b[0;36m_preprocess_data.<locals>.inner\u001b[1;34m(ax, data, *args, **kwargs)\u001b[0m\n\u001b[0;32m   1443\u001b[0m \u001b[38;5;129m@functools\u001b[39m\u001b[38;5;241m.\u001b[39mwraps(func)\n\u001b[0;32m   1444\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21minner\u001b[39m(ax, \u001b[38;5;241m*\u001b[39margs, data\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mNone\u001b[39;00m, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs):\n\u001b[0;32m   1445\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m data \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[1;32m-> 1446\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m func(ax, \u001b[38;5;241m*\u001b[39m\u001b[38;5;28mmap\u001b[39m(sanitize_sequence, args), \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n\u001b[0;32m   1448\u001b[0m     bound \u001b[38;5;241m=\u001b[39m new_sig\u001b[38;5;241m.\u001b[39mbind(ax, \u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n\u001b[0;32m   1449\u001b[0m     auto_label \u001b[38;5;241m=\u001b[39m (bound\u001b[38;5;241m.\u001b[39marguments\u001b[38;5;241m.\u001b[39mget(label_namer)\n\u001b[0;32m   1450\u001b[0m                   \u001b[38;5;129;01mor\u001b[39;00m bound\u001b[38;5;241m.\u001b[39mkwargs\u001b[38;5;241m.\u001b[39mget(label_namer))\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\matplotlib\\axes\\_axes.py:4572\u001b[0m, in \u001b[0;36mAxes.scatter\u001b[1;34m(self, x, y, s, c, marker, cmap, norm, vmin, vmax, alpha, linewidths, edgecolors, plotnonfinite, **kwargs)\u001b[0m\n\u001b[0;32m   4462\u001b[0m \u001b[38;5;250m\u001b[39m\u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[0;32m   4463\u001b[0m \u001b[38;5;124;03mA scatter plot of *y* vs. *x* with varying marker size and/or color.\u001b[39;00m\n\u001b[0;32m   4464\u001b[0m \n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m   4569\u001b[0m \n\u001b[0;32m   4570\u001b[0m \u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[0;32m   4571\u001b[0m \u001b[38;5;66;03m# Process **kwargs to handle aliases, conflicts with explicit kwargs:\u001b[39;00m\n\u001b[1;32m-> 4572\u001b[0m x, y \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_process_unit_info([(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mx\u001b[39m\u001b[38;5;124m\"\u001b[39m, x), (\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124my\u001b[39m\u001b[38;5;124m\"\u001b[39m, y)], kwargs)\n\u001b[0;32m   4573\u001b[0m \u001b[38;5;66;03m# np.ma.ravel yields an ndarray, not a masked array,\u001b[39;00m\n\u001b[0;32m   4574\u001b[0m \u001b[38;5;66;03m# unless its argument is a masked array.\u001b[39;00m\n\u001b[0;32m   4575\u001b[0m x \u001b[38;5;241m=\u001b[39m np\u001b[38;5;241m.\u001b[39mma\u001b[38;5;241m.\u001b[39mravel(x)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\matplotlib\\axes\\_base.py:2549\u001b[0m, in \u001b[0;36m_AxesBase._process_unit_info\u001b[1;34m(self, datasets, kwargs, convert)\u001b[0m\n\u001b[0;32m   2547\u001b[0m     \u001b[38;5;66;03m# Update from data if axis is already set but no unit is set yet.\u001b[39;00m\n\u001b[0;32m   2548\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m axis \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m \u001b[38;5;129;01mand\u001b[39;00m data \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m \u001b[38;5;129;01mand\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m axis\u001b[38;5;241m.\u001b[39mhave_units():\n\u001b[1;32m-> 2549\u001b[0m         axis\u001b[38;5;241m.\u001b[39mupdate_units(data)\n\u001b[0;32m   2550\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m axis_name, axis \u001b[38;5;129;01min\u001b[39;00m axis_map\u001b[38;5;241m.\u001b[39mitems():\n\u001b[0;32m   2551\u001b[0m     \u001b[38;5;66;03m# Return if no axis is set.\u001b[39;00m\n\u001b[0;32m   2552\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m axis \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\matplotlib\\axis.py:1675\u001b[0m, in \u001b[0;36mAxis.update_units\u001b[1;34m(self, data)\u001b[0m\n\u001b[0;32m   1673\u001b[0m neednew \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mconverter \u001b[38;5;241m!=\u001b[39m converter\n\u001b[0;32m   1674\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mconverter \u001b[38;5;241m=\u001b[39m converter\n\u001b[1;32m-> 1675\u001b[0m default \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mconverter\u001b[38;5;241m.\u001b[39mdefault_units(data, \u001b[38;5;28mself\u001b[39m)\n\u001b[0;32m   1676\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m default \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m \u001b[38;5;129;01mand\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39munits \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[0;32m   1677\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mset_units(default)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\matplotlib\\category.py:105\u001b[0m, in \u001b[0;36mStrCategoryConverter.default_units\u001b[1;34m(data, axis)\u001b[0m\n\u001b[0;32m    103\u001b[0m \u001b[38;5;66;03m# the conversion call stack is default_units -> axis_info -> convert\u001b[39;00m\n\u001b[0;32m    104\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m axis\u001b[38;5;241m.\u001b[39munits \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[1;32m--> 105\u001b[0m     axis\u001b[38;5;241m.\u001b[39mset_units(UnitData(data))\n\u001b[0;32m    106\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m    107\u001b[0m     axis\u001b[38;5;241m.\u001b[39munits\u001b[38;5;241m.\u001b[39mupdate(data)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\matplotlib\\category.py:181\u001b[0m, in \u001b[0;36mUnitData.__init__\u001b[1;34m(self, data)\u001b[0m\n\u001b[0;32m    179\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_counter \u001b[38;5;241m=\u001b[39m itertools\u001b[38;5;241m.\u001b[39mcount()\n\u001b[0;32m    180\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m data \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[1;32m--> 181\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mupdate(data)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\matplotlib\\category.py:216\u001b[0m, in \u001b[0;36mUnitData.update\u001b[1;34m(self, data)\u001b[0m\n\u001b[0;32m    213\u001b[0m convertible \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mTrue\u001b[39;00m\n\u001b[0;32m    214\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m val \u001b[38;5;129;01min\u001b[39;00m OrderedDict\u001b[38;5;241m.\u001b[39mfromkeys(data):\n\u001b[0;32m    215\u001b[0m     \u001b[38;5;66;03m# OrderedDict just iterates over unique values in data.\u001b[39;00m\n\u001b[1;32m--> 216\u001b[0m     _api\u001b[38;5;241m.\u001b[39mcheck_isinstance((\u001b[38;5;28mstr\u001b[39m, \u001b[38;5;28mbytes\u001b[39m), value\u001b[38;5;241m=\u001b[39mval)\n\u001b[0;32m    217\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m convertible:\n\u001b[0;32m    218\u001b[0m         \u001b[38;5;66;03m# this will only be called so long as convertible is True.\u001b[39;00m\n\u001b[0;32m    219\u001b[0m         convertible \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_str_is_convertible(val)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\matplotlib\\_api\\__init__.py:93\u001b[0m, in \u001b[0;36mcheck_isinstance\u001b[1;34m(_types, **kwargs)\u001b[0m\n\u001b[0;32m     91\u001b[0m     names\u001b[38;5;241m.\u001b[39mremove(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mNone\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m     92\u001b[0m     names\u001b[38;5;241m.\u001b[39mappend(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mNone\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[1;32m---> 93\u001b[0m \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mTypeError\u001b[39;00m(\n\u001b[0;32m     94\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;132;01m{!r}\u001b[39;00m\u001b[38;5;124m must be an instance of \u001b[39m\u001b[38;5;132;01m{}\u001b[39;00m\u001b[38;5;124m, not a \u001b[39m\u001b[38;5;132;01m{}\u001b[39;00m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;241m.\u001b[39mformat(\n\u001b[0;32m     95\u001b[0m         k,\n\u001b[0;32m     96\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m, \u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;241m.\u001b[39mjoin(names[:\u001b[38;5;241m-\u001b[39m\u001b[38;5;241m1\u001b[39m]) \u001b[38;5;241m+\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m or \u001b[39m\u001b[38;5;124m\"\u001b[39m \u001b[38;5;241m+\u001b[39m names[\u001b[38;5;241m-\u001b[39m\u001b[38;5;241m1\u001b[39m]\n\u001b[0;32m     97\u001b[0m         \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mlen\u001b[39m(names) \u001b[38;5;241m>\u001b[39m \u001b[38;5;241m1\u001b[39m \u001b[38;5;28;01melse\u001b[39;00m names[\u001b[38;5;241m0\u001b[39m],\n\u001b[0;32m     98\u001b[0m         type_name(\u001b[38;5;28mtype\u001b[39m(v))))\n",
      "\u001b[1;31mTypeError\u001b[0m: 'value' must be an instance of str or bytes, not a float"
     ]
    },
    {
     "data": {
      "image/png": "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N8ftt9/eebutrS327NkTEyZM6LV5+7M8ZxER8dRTT0Vzc3PceeedvTnmgJHnPF5++eW44IIL4u67747q6uqYMWNGNDY2xqhRo3p77H4pz1nMmjUrRo4cGTNnzoyLL7447rrrrnjggQeitLS0t8futy6//PLYvn17zJw581P39cTzd5+FindiTkees/hv77zzTixYsCAmTJgQs2fP/kJnHCjynEVTU1OsWrUqVq5cGYWFhb0240CS5zzee++9eOyxx6KmpiZefPHFqK+vjwcffDC2bt3aa/P2Z3nO4tSpUzF+/Ph44oknYt++fVFfXx9Lly6N119/vdfm7e9GjRrVrStUPfH83Weh4p2Y05HnLM7Yt29fzJ8/Py688MJ45JFHXFLtId09iw8++CAWLVoU9957b5x//vm9OuNAkudrY8iQITF16tS46qqroqioKKZMmRJz586NLVu29Nq8/Vmes7j//vujvLw8Jk2aFEOGDInrr78+KioqYtOmTb02L/9PTzx/91moeCfmdOQ5i4iIjRs3xq233hq33HJLrFy5MoYMGdKb4/Zr3T2LV155JQ4ePBhLly6NqqqqqKqqioiIO+64I+67777eHrvfyvO1UVZWFh9++GGXtY8++igy71LSI/KcRUtLy1lnUVRUFIMHD+6VWfn/euT5uyd+8/fz+uEPf5gtWrQoe//99zt/g3v16tVn7Ttw4EA2ceLE7Nlnn+38reGJEydmb775Zh9M3T919yy2bt2aXXzxxdnOnTv7YMqBobtn8X951c8Xo7vn8ec//zn71re+lT311FPZ6dOns8bGxqyioiJ77rnn+mDq/qm7Z7Fq1aqsuro6+/vf/5599NFH2ZYtW7KJEydm//jHP/pg6v7v07739MTzd5+GyjvvvJMtXLgwu+yyy7Jvf/vb2QMPPJB1dHRkWZZlFRUV2dNPP925d+fOnVlNTU1WUVGRzZo1K3vhhRf6aux+qbtnMXv27Gz8+PFZRUVFl4+f//znfTl+v5Ln6+K/CZUvRp7zeOGFF7Lvfe97WWVlZTZ16tRsw4YNfTV2v9Tdszh16lS2evXq7Lvf/W526aWXZtddd53/XH2B/u/3np5+/vbuyQBAsvwJfQAgWUIFAEiWUAEAkiVUAIBkCRUAIFlCBQBIllABAJIlVACAZAkVACBZQgUASJZQAQCS9b/q9LHmV4SDeQAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "for i, col1 in enumerate(df.columns):\n",
    "    for j, col2 in enumerate(df.columns):\n",
    "        if i < j:\n",
    "            plt.scatter(df[col1], df[col2])\n",
    "            plt.title(col1 + ' vs ' + col2)\n",
    "            plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "920eb058",
   "metadata": {},
   "outputs": [],
   "source": [
    "It imports the matplotlib.pyplot module as plt.\n",
    "It iterates over all pairs of columns in the DataFrame df using nested loops.\n",
    "For each pair of columns, it creates a scatter plot using plt.scatter(df[col1], df[col2]).\n",
    "It sets the title of the scatter plot to indicate which columns are being compared (plt.title(col1 + ' vs ' + col2)).\n",
    "Finally, it displays each scatter plot using plt.show().\n",
    "The purpose of this code is to visualize the relationship between different pairs of columns in the DataFrame df. Each scatter\n",
    "plot shows how the values of one column (on the x-axis) vary with the values of another column (on the y-axis). This \n",
    "visualization helps in understanding correlations or patterns between different variables in the dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 125,
   "id": "dad15c22",
   "metadata": {},
   "outputs": [],
   "source": [
    "FEATURE TRANSFORMATION"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "246f0b91",
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
   "execution_count": null,
   "id": "f338441b",
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
   "execution_count": 20,
   "id": "cb7ab4f4",
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
   "execution_count": 128,
   "id": "10d22f29",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 4 entries, 0 to 3\n",
      "Data columns (total 3 columns):\n",
      " #   Column    Non-Null Count  Dtype  \n",
      "---  ------    --------------  -----  \n",
      " 0   feature1  4 non-null      float64\n",
      " 1   feature2  4 non-null      int64  \n",
      " 2   target    4 non-null      int64  \n",
      "dtypes: float64(1), int64(2)\n",
      "memory usage: 228.0 bytes\n",
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
   "execution_count": 22,
   "id": "45d7519b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Ind_ID                     0\n",
      "Test_date                  0\n",
      "Cough                      0\n",
      "Fever                      0\n",
      "Sore_throat                0\n",
      "Shortness_of_breath        0\n",
      "Headache                   0\n",
      "Corona                     0\n",
      "Sex                    19030\n",
      "Known_contact              0\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "\n",
    " \n",
    "\n",
    "print(covid_df.isnull().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f2f2300f",
   "metadata": {},
   "outputs": [],
   "source": [
    "MISSING VALUES IMPUTATION , ENCODING  AND KNN IMPUTATION"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "f5b31bce",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.impute import KNNImputer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "b0799451",
   "metadata": {},
   "outputs": [],
   "source": [
    "# creating a copy of original dataframe\n",
    "\n",
    "imputed_data = covid_df.copy(deep=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "e4387855",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import KNNImputer\n",
    "from sklearn.impute import KNNImputer\n",
    "\n",
    "# Column indices to impute\n",
    "cols = [2, 3, 4, 5, 6, 8, 9] \n",
    "\n",
    "# Create imputer \n",
    "knn = KNNImputer(n_neighbors=5, weights='uniform')\n",
    "\n",
    "# Transform data\n",
    "imputed = knn.fit_transform(covid_df.iloc[:, cols])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "42f708ea",
   "metadata": {},
   "outputs": [],
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
   "execution_count": 32,
   "id": "54862e71",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1 = pd.DataFrame(imputed,columns=columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "78e400e5",
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
   "execution_count": 35,
   "id": "48439a70",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1['Sex'] = df1['Sex'].apply(change)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "59b2ea95",
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
   "execution_count": 33,
   "id": "3a63e0f8",
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
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "imputed_data.head()\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "5021523c",
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
   "execution_count": 35,
   "id": "557b30ed",
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
     "execution_count": 35,
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
   "id": "ca70e126",
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
      " 0   Sex     5 non-null      uint8\n",
      "dtypes: uint8(1)\n",
      "memory usage: 137.0 bytes\n"
     ]
    }
   ],
   "source": [
    "imputed_data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "9cd250de",
   "metadata": {},
   "outputs": [],
   "source": [
    "imputed_data['Sex'] = imputed_data['Sex'].astype('uint8')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "e30cca41",
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
   "execution_count": 131,
   "id": "2de55e2f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "    Cough_symptoms  Fever  Sore_throat  Shortness_of_breath  Headache  \\\n",
      "0                1      0            0                    0         0   \n",
      "1                1      0            0                    0         0   \n",
      "2                0      1            0                    0         0   \n",
      "3                0      1            0                    0         0   \n",
      "4                0      0            1                    0         1   \n",
      "..             ...    ...          ...                  ...       ...   \n",
      "95               1      1            1                    0         0   \n",
      "96               1      0            1                    1         0   \n",
      "97               1      0            1                    0         0   \n",
      "98               0      1            1                    0         1   \n",
      "99               1      1            0                    1         1   \n",
      "\n",
      "    Known_contact  \n",
      "0               0  \n",
      "1               0  \n",
      "2               1  \n",
      "3               0  \n",
      "4               1  \n",
      "..            ...  \n",
      "95              1  \n",
      "96              0  \n",
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
   "execution_count": 130,
   "id": "63ef433f",
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
   "execution_count": null,
   "id": "5812c356",
   "metadata": {},
   "outputs": [],
   "source": [
    "imputed_data['Sex'] = imputed_data['Sex'].astype('uint8')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "id": "fdf37618",
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
   "execution_count": 136,
   "id": "3ae50a34",
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
      "memory usage: 13.6 MB\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "\n",
    "\n",
    "# Change dtypes to category\n",
    "df = pd.DataFrame(data).astype('category') \n",
    "\n",
    "# Print dataframe info\n",
    "print(df.info())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "36686963",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   Patient_ID  Age  Gender Test_Result\n",
      "0           1   30    Male    Positive\n",
      "1           2   45  Female    Negative\n",
      "2           3   50    Male    Positive\n",
      "3           4   25  Female    Negative\n",
      "4           5   60    Male    Positive\n",
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 5 entries, 0 to 4\n",
      "Data columns (total 4 columns):\n",
      " #   Column       Non-Null Count  Dtype \n",
      "---  ------       --------------  ----- \n",
      " 0   Patient_ID   5 non-null      int64 \n",
      " 1   Age          5 non-null      int64 \n",
      " 2   Gender       5 non-null      object\n",
      " 3   Test_Result  5 non-null      object\n",
      "dtypes: int64(2), object(2)\n",
      "memory usage: 292.0+ bytes\n",
      "None\n",
      "       Patient_ID       Age\n",
      "count    5.000000   5.00000\n",
      "mean     3.000000  42.00000\n",
      "std      1.581139  14.40486\n",
      "min      1.000000  25.00000\n",
      "25%      2.000000  30.00000\n",
      "50%      3.000000  45.00000\n",
      "75%      4.000000  50.00000\n",
      "max      5.000000  60.00000\n"
     ]
    },
    {
     "ename": "KeyError",
     "evalue": "'column_name'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\pandas\\core\\indexes\\base.py:3653\u001b[0m, in \u001b[0;36mIndex.get_loc\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m   3652\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[1;32m-> 3653\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_engine\u001b[38;5;241m.\u001b[39mget_loc(casted_key)\n\u001b[0;32m   3654\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mKeyError\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m err:\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\pandas\\_libs\\index.pyx:147\u001b[0m, in \u001b[0;36mpandas._libs.index.IndexEngine.get_loc\u001b[1;34m()\u001b[0m\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\pandas\\_libs\\index.pyx:176\u001b[0m, in \u001b[0;36mpandas._libs.index.IndexEngine.get_loc\u001b[1;34m()\u001b[0m\n",
      "File \u001b[1;32mpandas\\_libs\\hashtable_class_helper.pxi:7080\u001b[0m, in \u001b[0;36mpandas._libs.hashtable.PyObjectHashTable.get_item\u001b[1;34m()\u001b[0m\n",
      "File \u001b[1;32mpandas\\_libs\\hashtable_class_helper.pxi:7088\u001b[0m, in \u001b[0;36mpandas._libs.hashtable.PyObjectHashTable.get_item\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;31mKeyError\u001b[0m: 'column_name'",
      "\nThe above exception was the direct cause of the following exception:\n",
      "\u001b[1;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[19], line 13\u001b[0m\n\u001b[0;32m     10\u001b[0m \u001b[38;5;28mprint\u001b[39m(data\u001b[38;5;241m.\u001b[39mdescribe())\n\u001b[0;32m     12\u001b[0m \u001b[38;5;66;03m# Accessing specific columns\u001b[39;00m\n\u001b[1;32m---> 13\u001b[0m \u001b[38;5;28mprint\u001b[39m(data[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mcolumn_name\u001b[39m\u001b[38;5;124m'\u001b[39m])  \u001b[38;5;66;03m# Replace 'column_name' with the actual name of the column\u001b[39;00m\n\u001b[0;32m     15\u001b[0m \u001b[38;5;66;03m# Filtering data based on conditions\u001b[39;00m\n\u001b[0;32m     16\u001b[0m filtered_data \u001b[38;5;241m=\u001b[39m data[data[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mcolumn_name\u001b[39m\u001b[38;5;124m'\u001b[39m] \u001b[38;5;241m>\u001b[39m threshold]  \u001b[38;5;66;03m# Replace 'column_name' and 'threshold' with actual values\u001b[39;00m\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\pandas\\core\\frame.py:3761\u001b[0m, in \u001b[0;36mDataFrame.__getitem__\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m   3759\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mcolumns\u001b[38;5;241m.\u001b[39mnlevels \u001b[38;5;241m>\u001b[39m \u001b[38;5;241m1\u001b[39m:\n\u001b[0;32m   3760\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_getitem_multilevel(key)\n\u001b[1;32m-> 3761\u001b[0m indexer \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mcolumns\u001b[38;5;241m.\u001b[39mget_loc(key)\n\u001b[0;32m   3762\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m is_integer(indexer):\n\u001b[0;32m   3763\u001b[0m     indexer \u001b[38;5;241m=\u001b[39m [indexer]\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\pandas\\core\\indexes\\base.py:3655\u001b[0m, in \u001b[0;36mIndex.get_loc\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m   3653\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_engine\u001b[38;5;241m.\u001b[39mget_loc(casted_key)\n\u001b[0;32m   3654\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mKeyError\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m err:\n\u001b[1;32m-> 3655\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mKeyError\u001b[39;00m(key) \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01merr\u001b[39;00m\n\u001b[0;32m   3656\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mTypeError\u001b[39;00m:\n\u001b[0;32m   3657\u001b[0m     \u001b[38;5;66;03m# If we have a listlike key, _check_indexing_error will raise\u001b[39;00m\n\u001b[0;32m   3658\u001b[0m     \u001b[38;5;66;03m#  InvalidIndexError. Otherwise we fall through and re-raise\u001b[39;00m\n\u001b[0;32m   3659\u001b[0m     \u001b[38;5;66;03m#  the TypeError.\u001b[39;00m\n\u001b[0;32m   3660\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_check_indexing_error(key)\n",
      "\u001b[1;31mKeyError\u001b[0m: 'column_name'"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "# Display the first few rows of the DataFrame\n",
    "print(data.head())\n",
    "\n",
    "# Get some basic information about the DataFrame\n",
    "print(data.info())\n",
    "\n",
    "# Summary statistics of numerical columns\n",
    "print(data.describe())\n",
    "\n",
    "# Accessing specific columns\n",
    "print(data['column_name'])  # Replace 'column_name' with the actual name of the column\n",
    "\n",
    "# Filtering data based on conditions\n",
    "filtered_data = data[data['column_name'] > threshold]  # Replace 'column_name' and 'threshold' with actual values\n",
    "\n",
    "# Grouping data and performing aggregations\n",
    "grouped_data = data.groupby('column_name').agg({'another_column': 'mean'})  # Replace 'column_name' and 'another_column' with actual names\n",
    "\n",
    "# Handling missing values\n",
    "clean_data = data.dropna()  # Drop rows with missing values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a0eea895",
   "metadata": {},
   "outputs": [],
   "source": [
    "FEATURE ENGINEERING                 "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8e234b04",
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
   "execution_count": 70,
   "id": "26d89779",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.feature_selection import SelectKBest\n",
    "from sklearn.feature_selection import chi2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "60cca4ea",
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
   "id": "bedc6288",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "This shows covid symptoms are the important features of the dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "id": "c122b169",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "       Cough_symptoms      Fever  Sore_throat  Shortness_of_breath   Headache  \\\n",
      "count       10.000000  10.000000    10.000000            10.000000  10.000000   \n",
      "mean         0.500000   0.500000     0.500000             0.300000   0.700000   \n",
      "std          0.527046   0.527046     0.527046             0.483046   0.483046   \n",
      "min          0.000000   0.000000     0.000000             0.000000   0.000000   \n",
      "25%          0.000000   0.000000     0.000000             0.000000   0.250000   \n",
      "50%          0.500000   0.500000     0.500000             0.000000   1.000000   \n",
      "75%          1.000000   1.000000     1.000000             0.750000   1.000000   \n",
      "max          1.000000   1.000000     1.000000             1.000000   1.000000   \n",
      "\n",
      "       Known_contact  \n",
      "count      10.000000  \n",
      "mean        0.500000  \n",
      "std         0.527046  \n",
      "min         0.000000  \n",
      "25%         0.000000  \n",
      "50%         0.500000  \n",
      "75%         1.000000  \n",
      "max         1.000000  \n",
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
   "execution_count": 77,
   "id": "e8cab5b7",
   "metadata": {},
   "outputs": [],
   "source": [
    "FEATURES = ['Cough_symptoms', 'Fever', 'Sore_throat', 'Shortness_of_breath',\n",
    "            'Headache', 'Known_contact']\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "ec1570ed",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   Cough_symptoms  Fever  Sore_throat  Shortness_of_breath  Headache  \\\n",
      "0               1      0            0                    0         1   \n",
      "1               0      1            1                    0         1   \n",
      "2               1      0            0                    1         1   \n",
      "3               0      0            0                    0         0   \n",
      "4               1      0            1                    0         1   \n",
      "5               0      1            0                    0         1   \n",
      "6               0      0            1                    0         0   \n",
      "7               1      1            1                    1         1   \n",
      "8               0      1            1                    1         1   \n",
      "9               1      1            0                    0         0   \n",
      "\n",
      "   Known_contact  \n",
      "0              0  \n",
      "1              1  \n",
      "2              1  \n",
      "3              0  \n",
      "4              0  \n",
      "5              1  \n",
      "6              1  \n",
      "7              0  \n",
      "8              0  \n",
      "9              1  \n"
     ]
    }
   ],
   "source": [
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "# Assume df is your DataFrame\n",
    "df = pd.DataFrame(np.random.randint(0, 2, size=(10, len(FEATURES))), columns=FEATURES)\n",
    "print(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 106,
   "id": "47e3e7e5",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "df = pd.DataFrame({\n",
    "    'feature_1': [0.1, 0.2, 0.3, 0.4, 0.5], \n",
    "    'feature_2': [1, 2, 3, 4, 5],\n",
    "    'target': [1, 0, 1, 0, 1]\n",
    "})\n",
    "\n",
    "features = df[['feature_1', 'feature_2']] \n",
    "target = df['target']\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 137,
   "id": "7cce14c7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Corona\n",
      "0    54632\n",
      "1     3055\n",
      "Name: count, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# Split data into train and test sets\n",
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(df, df['Corona'], test_size=0.3, random_state=42)\n",
    "\n",
    "# Print value counts for y_train\n",
    "print(y_train.value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 139,
   "id": "05d1e765",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Corona\n",
      "0    127341\n",
      "1      7262\n",
      "Name: count, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "\n",
    "\n",
    "# Split data into train and test sets \n",
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(df, df['Corona'], test_size=0.3, random_state=42)\n",
    "\n",
    "# Print value counts for y_train\n",
    "print(y_train.value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b3323b9b",
   "metadata": {},
   "outputs": [],
   "source": [
    "MODEL TRAINING"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "3b4f6a56",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.impute import KNNImputer\n",
    "from sklearn.ensemble import RandomForestClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "78f7ddeb",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Dell\\AppData\\Local\\Temp\\ipykernel_11196\\3103726082.py:12: DtypeWarning: Columns (2,3,4,5,6,8) have mixed types. Specify dtype option on import or set low_memory=False.\n",
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
   "execution_count": 6,
   "id": "34a79534",
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
   "execution_count": 32,
   "id": "d809f46e",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2020-03-11 00:00:00\n"
     ]
    }
   ],
   "source": [
    "from datetime import datetime\n",
    "\n",
    "date_string = \"11-03-2020\"\n",
    "date_format = \"%d-%m-%Y\"\n",
    "\n",
    "try:\n",
    "    parsed_date = datetime.strptime(date_string, date_format)\n",
    "    print(parsed_date)\n",
    "except ValueError:\n",
    "    print(\"Date string doesn't match the specified format.\")\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e9bbc139",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# Impute missing values\n",
    "imputer = KNNImputer()\n",
    "df_imputed = pd.DataFrame(imputer.fit_transform(df))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "8c783e5f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6], 'target': [7, 8, 9]}) \n",
    "\n",
    "X = df.drop('target', axis=1)\n",
    "y = df['target']\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "c2f807fe",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>RandomForestClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">RandomForestClassifier</label><div class=\"sk-toggleable__content\"><pre>RandomForestClassifier()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "RandomForestClassifier()"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "# Train model (e.g. Random Forest)\n",
    "model = RandomForestClassifier()\n",
    "model.fit(X_train, y_train)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "b05005f1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy:96.75164723155889\n"
     ]
    }
   ],
   "source": [
    "\n",
    "\n",
    "X = df.drop('target', axis=1) \n",
    "y = df['target']\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Train model, make predictions, calculate accuracy (code not shown)\n",
    "\n",
    "\n",
    "\n",
    "print(\"Accuracy:, accuracy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "47daafa3",
   "metadata": {},
   "outputs": [],
   "source": [
    "RANDOM FOREST"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "595a942d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "72012d20",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Dell\\AppData\\Local\\Temp\\ipykernel_7336\\3103726082.py:12: DtypeWarning: Columns (2,3,4,5,6,8) have mixed types. Specify dtype option on import or set low_memory=False.\n",
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
   "execution_count": 13,
   "id": "e2b8e78d",
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
   "execution_count": 15,
   "id": "dc39bbf0",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6], 'target': [7, 8, 9]}) \n",
    "\n",
    "X = df.drop('target', axis=1)\n",
    "y = df['target']\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "551cea7c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-2 {color: black;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-2\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>RandomForestClassifier(random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" checked><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">RandomForestClassifier</label><div class=\"sk-toggleable__content\"><pre>RandomForestClassifier(random_state=42)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "RandomForestClassifier(random_state=42)"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "# Train a Random Forest model\n",
    "rf = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "rf.fit(X_train, y_train)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "3108fc82",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 96.75164723155889\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "# Split original data \n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Split X_train and y_train into training and validation sets\n",
    "X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)\n",
    "\n",
    "# Train model on new X_train and y_train\n",
    "rf = RandomForestClassifier()\n",
    "rf.fit(X_train, y_train)\n",
    "\n",
    "# Calculate accuracy\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(\"Accuracy:\", accuracy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "8ed8c62b",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Dell\\anaconda3\\Lib\\site-packages\\sklearn\\base.py:464: UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "ename": "ValueError",
     "evalue": "X has 6 features, but RandomForestClassifier is expecting 2 features as input.",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[23], line 9\u001b[0m\n\u001b[0;32m      6\u001b[0m new_data \u001b[38;5;241m=\u001b[39m [[\u001b[38;5;241m1\u001b[39m, \u001b[38;5;241m0\u001b[39m, \u001b[38;5;241m1\u001b[39m, \u001b[38;5;241m0\u001b[39m, \u001b[38;5;241m1\u001b[39m, \u001b[38;5;241m0\u001b[39m]] \n\u001b[0;32m      8\u001b[0m \u001b[38;5;66;03m# Make prediction\u001b[39;00m\n\u001b[1;32m----> 9\u001b[0m prediction \u001b[38;5;241m=\u001b[39m rf\u001b[38;5;241m.\u001b[39mpredict(new_data)\n\u001b[0;32m     11\u001b[0m \u001b[38;5;28mprint\u001b[39m(prediction)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\sklearn\\ensemble\\_forest.py:823\u001b[0m, in \u001b[0;36mForestClassifier.predict\u001b[1;34m(self, X)\u001b[0m\n\u001b[0;32m    802\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mpredict\u001b[39m(\u001b[38;5;28mself\u001b[39m, X):\n\u001b[0;32m    803\u001b[0m \u001b[38;5;250m    \u001b[39m\u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[0;32m    804\u001b[0m \u001b[38;5;124;03m    Predict class for X.\u001b[39;00m\n\u001b[0;32m    805\u001b[0m \n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    821\u001b[0m \u001b[38;5;124;03m        The predicted classes.\u001b[39;00m\n\u001b[0;32m    822\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n\u001b[1;32m--> 823\u001b[0m     proba \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mpredict_proba(X)\n\u001b[0;32m    825\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mn_outputs_ \u001b[38;5;241m==\u001b[39m \u001b[38;5;241m1\u001b[39m:\n\u001b[0;32m    826\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mclasses_\u001b[38;5;241m.\u001b[39mtake(np\u001b[38;5;241m.\u001b[39margmax(proba, axis\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m1\u001b[39m), axis\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m0\u001b[39m)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\sklearn\\ensemble\\_forest.py:865\u001b[0m, in \u001b[0;36mForestClassifier.predict_proba\u001b[1;34m(self, X)\u001b[0m\n\u001b[0;32m    863\u001b[0m check_is_fitted(\u001b[38;5;28mself\u001b[39m)\n\u001b[0;32m    864\u001b[0m \u001b[38;5;66;03m# Check data\u001b[39;00m\n\u001b[1;32m--> 865\u001b[0m X \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_validate_X_predict(X)\n\u001b[0;32m    867\u001b[0m \u001b[38;5;66;03m# Assign chunk of trees to jobs\u001b[39;00m\n\u001b[0;32m    868\u001b[0m n_jobs, _, _ \u001b[38;5;241m=\u001b[39m _partition_estimators(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mn_estimators, \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mn_jobs)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\sklearn\\ensemble\\_forest.py:599\u001b[0m, in \u001b[0;36mBaseForest._validate_X_predict\u001b[1;34m(self, X)\u001b[0m\n\u001b[0;32m    596\u001b[0m \u001b[38;5;250m\u001b[39m\u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[0;32m    597\u001b[0m \u001b[38;5;124;03mValidate X whenever one tries to predict, apply, predict_proba.\"\"\"\u001b[39;00m\n\u001b[0;32m    598\u001b[0m check_is_fitted(\u001b[38;5;28mself\u001b[39m)\n\u001b[1;32m--> 599\u001b[0m X \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_validate_data(X, dtype\u001b[38;5;241m=\u001b[39mDTYPE, accept_sparse\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcsr\u001b[39m\u001b[38;5;124m\"\u001b[39m, reset\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mFalse\u001b[39;00m)\n\u001b[0;32m    600\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m issparse(X) \u001b[38;5;129;01mand\u001b[39;00m (X\u001b[38;5;241m.\u001b[39mindices\u001b[38;5;241m.\u001b[39mdtype \u001b[38;5;241m!=\u001b[39m np\u001b[38;5;241m.\u001b[39mintc \u001b[38;5;129;01mor\u001b[39;00m X\u001b[38;5;241m.\u001b[39mindptr\u001b[38;5;241m.\u001b[39mdtype \u001b[38;5;241m!=\u001b[39m np\u001b[38;5;241m.\u001b[39mintc):\n\u001b[0;32m    601\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mNo support for np.int64 index based sparse matrices\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\sklearn\\base.py:625\u001b[0m, in \u001b[0;36mBaseEstimator._validate_data\u001b[1;34m(self, X, y, reset, validate_separately, cast_to_ndarray, **check_params)\u001b[0m\n\u001b[0;32m    622\u001b[0m     out \u001b[38;5;241m=\u001b[39m X, y\n\u001b[0;32m    624\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m no_val_X \u001b[38;5;129;01mand\u001b[39;00m check_params\u001b[38;5;241m.\u001b[39mget(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mensure_2d\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;28;01mTrue\u001b[39;00m):\n\u001b[1;32m--> 625\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_check_n_features(X, reset\u001b[38;5;241m=\u001b[39mreset)\n\u001b[0;32m    627\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m out\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\sklearn\\base.py:414\u001b[0m, in \u001b[0;36mBaseEstimator._check_n_features\u001b[1;34m(self, X, reset)\u001b[0m\n\u001b[0;32m    411\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m\n\u001b[0;32m    413\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m n_features \u001b[38;5;241m!=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mn_features_in_:\n\u001b[1;32m--> 414\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\n\u001b[0;32m    415\u001b[0m         \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mX has \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mn_features\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m features, but \u001b[39m\u001b[38;5;132;01m{\u001b[39;00m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m\u001b[38;5;18m__class__\u001b[39m\u001b[38;5;241m.\u001b[39m\u001b[38;5;18m__name__\u001b[39m\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    416\u001b[0m         \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mis expecting \u001b[39m\u001b[38;5;132;01m{\u001b[39;00m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mn_features_in_\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m features as input.\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    417\u001b[0m     )\n",
      "\u001b[1;31mValueError\u001b[0m: X has 6 features, but RandomForestClassifier is expecting 2 features as input."
     ]
    }
   ],
   "source": [
    "# Train model \n",
    "rf = RandomForestClassifier()\n",
    "rf.fit(X_train, y_train)\n",
    "\n",
    "# New data point \n",
    "new_data = [[1, 0, 1, 0, 1, 0]] \n",
    "\n",
    "# Make prediction\n",
    "prediction = rf.predict(new_data)\n",
    "\n",
    "print(prediction)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2ed0a6e0",
   "metadata": {},
   "outputs": [],
   "source": [
    "HYPER PARAMETER TUNING"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "16874818",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import RandomizedSearchCV\n",
    "from sklearn.ensemble import RandomForestClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "f3dd78c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import RandomizedSearchCV\n",
    "\n",
    "# Rest of code"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "5c981133",
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "Cannot have number of splits n_splits=5 greater than the number of samples: n_samples=3.",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[30], line 27\u001b[0m\n\u001b[0;32m     18\u001b[0m rf_search \u001b[38;5;241m=\u001b[39m RandomizedSearchCV(\n\u001b[0;32m     19\u001b[0m     RandomForestClassifier(),\n\u001b[0;32m     20\u001b[0m     rf_grid,\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m     23\u001b[0m     scoring\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124maccuracy\u001b[39m\u001b[38;5;124m'\u001b[39m\n\u001b[0;32m     24\u001b[0m )\n\u001b[0;32m     26\u001b[0m \u001b[38;5;66;03m# Fit on data\u001b[39;00m\n\u001b[1;32m---> 27\u001b[0m rf_search\u001b[38;5;241m.\u001b[39mfit(X, y)\n\u001b[0;32m     29\u001b[0m \u001b[38;5;66;03m# Get best hyperparameters \u001b[39;00m\n\u001b[0;32m     30\u001b[0m \u001b[38;5;28mprint\u001b[39m(rf_search\u001b[38;5;241m.\u001b[39mbest_params_)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\sklearn\\base.py:1151\u001b[0m, in \u001b[0;36m_fit_context.<locals>.decorator.<locals>.wrapper\u001b[1;34m(estimator, *args, **kwargs)\u001b[0m\n\u001b[0;32m   1144\u001b[0m     estimator\u001b[38;5;241m.\u001b[39m_validate_params()\n\u001b[0;32m   1146\u001b[0m \u001b[38;5;28;01mwith\u001b[39;00m config_context(\n\u001b[0;32m   1147\u001b[0m     skip_parameter_validation\u001b[38;5;241m=\u001b[39m(\n\u001b[0;32m   1148\u001b[0m         prefer_skip_nested_validation \u001b[38;5;129;01mor\u001b[39;00m global_skip_validation\n\u001b[0;32m   1149\u001b[0m     )\n\u001b[0;32m   1150\u001b[0m ):\n\u001b[1;32m-> 1151\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m fit_method(estimator, \u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\sklearn\\model_selection\\_search.py:898\u001b[0m, in \u001b[0;36mBaseSearchCV.fit\u001b[1;34m(self, X, y, groups, **fit_params)\u001b[0m\n\u001b[0;32m    892\u001b[0m     results \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_format_results(\n\u001b[0;32m    893\u001b[0m         all_candidate_params, n_splits, all_out, all_more_results\n\u001b[0;32m    894\u001b[0m     )\n\u001b[0;32m    896\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m results\n\u001b[1;32m--> 898\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_run_search(evaluate_candidates)\n\u001b[0;32m    900\u001b[0m \u001b[38;5;66;03m# multimetric is determined here because in the case of a callable\u001b[39;00m\n\u001b[0;32m    901\u001b[0m \u001b[38;5;66;03m# self.scoring the return type is only known after calling\u001b[39;00m\n\u001b[0;32m    902\u001b[0m first_test_score \u001b[38;5;241m=\u001b[39m all_out[\u001b[38;5;241m0\u001b[39m][\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mtest_scores\u001b[39m\u001b[38;5;124m\"\u001b[39m]\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\sklearn\\model_selection\\_search.py:1806\u001b[0m, in \u001b[0;36mRandomizedSearchCV._run_search\u001b[1;34m(self, evaluate_candidates)\u001b[0m\n\u001b[0;32m   1804\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m_run_search\u001b[39m(\u001b[38;5;28mself\u001b[39m, evaluate_candidates):\n\u001b[0;32m   1805\u001b[0m \u001b[38;5;250m    \u001b[39m\u001b[38;5;124;03m\"\"\"Search n_iter candidates from param_distributions\"\"\"\u001b[39;00m\n\u001b[1;32m-> 1806\u001b[0m     evaluate_candidates(\n\u001b[0;32m   1807\u001b[0m         ParameterSampler(\n\u001b[0;32m   1808\u001b[0m             \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mparam_distributions, \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mn_iter, random_state\u001b[38;5;241m=\u001b[39m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mrandom_state\n\u001b[0;32m   1809\u001b[0m         )\n\u001b[0;32m   1810\u001b[0m     )\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\sklearn\\model_selection\\_search.py:857\u001b[0m, in \u001b[0;36mBaseSearchCV.fit.<locals>.evaluate_candidates\u001b[1;34m(candidate_params, cv, more_results)\u001b[0m\n\u001b[0;32m    837\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mverbose \u001b[38;5;241m>\u001b[39m \u001b[38;5;241m0\u001b[39m:\n\u001b[0;32m    838\u001b[0m     \u001b[38;5;28mprint\u001b[39m(\n\u001b[0;32m    839\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mFitting \u001b[39m\u001b[38;5;132;01m{0}\u001b[39;00m\u001b[38;5;124m folds for each of \u001b[39m\u001b[38;5;132;01m{1}\u001b[39;00m\u001b[38;5;124m candidates,\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    840\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m totalling \u001b[39m\u001b[38;5;132;01m{2}\u001b[39;00m\u001b[38;5;124m fits\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;241m.\u001b[39mformat(\n\u001b[0;32m    841\u001b[0m             n_splits, n_candidates, n_candidates \u001b[38;5;241m*\u001b[39m n_splits\n\u001b[0;32m    842\u001b[0m         )\n\u001b[0;32m    843\u001b[0m     )\n\u001b[0;32m    845\u001b[0m out \u001b[38;5;241m=\u001b[39m parallel(\n\u001b[0;32m    846\u001b[0m     delayed(_fit_and_score)(\n\u001b[0;32m    847\u001b[0m         clone(base_estimator),\n\u001b[0;32m    848\u001b[0m         X,\n\u001b[0;32m    849\u001b[0m         y,\n\u001b[0;32m    850\u001b[0m         train\u001b[38;5;241m=\u001b[39mtrain,\n\u001b[0;32m    851\u001b[0m         test\u001b[38;5;241m=\u001b[39mtest,\n\u001b[0;32m    852\u001b[0m         parameters\u001b[38;5;241m=\u001b[39mparameters,\n\u001b[0;32m    853\u001b[0m         split_progress\u001b[38;5;241m=\u001b[39m(split_idx, n_splits),\n\u001b[0;32m    854\u001b[0m         candidate_progress\u001b[38;5;241m=\u001b[39m(cand_idx, n_candidates),\n\u001b[0;32m    855\u001b[0m         \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mfit_and_score_kwargs,\n\u001b[0;32m    856\u001b[0m     )\n\u001b[1;32m--> 857\u001b[0m     \u001b[38;5;28;01mfor\u001b[39;00m (cand_idx, parameters), (split_idx, (train, test)) \u001b[38;5;129;01min\u001b[39;00m product(\n\u001b[0;32m    858\u001b[0m         \u001b[38;5;28menumerate\u001b[39m(candidate_params), \u001b[38;5;28menumerate\u001b[39m(cv\u001b[38;5;241m.\u001b[39msplit(X, y, groups))\n\u001b[0;32m    859\u001b[0m     )\n\u001b[0;32m    860\u001b[0m )\n\u001b[0;32m    862\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mlen\u001b[39m(out) \u001b[38;5;241m<\u001b[39m \u001b[38;5;241m1\u001b[39m:\n\u001b[0;32m    863\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\n\u001b[0;32m    864\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mNo fits were performed. \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    865\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mWas the CV iterator empty? \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    866\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mWere there no candidates?\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    867\u001b[0m     )\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\sklearn\\model_selection\\_split.py:370\u001b[0m, in \u001b[0;36m_BaseKFold.split\u001b[1;34m(self, X, y, groups)\u001b[0m\n\u001b[0;32m    368\u001b[0m n_samples \u001b[38;5;241m=\u001b[39m _num_samples(X)\n\u001b[0;32m    369\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mn_splits \u001b[38;5;241m>\u001b[39m n_samples:\n\u001b[1;32m--> 370\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\n\u001b[0;32m    371\u001b[0m         (\n\u001b[0;32m    372\u001b[0m             \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mCannot have number of splits n_splits=\u001b[39m\u001b[38;5;132;01m{0}\u001b[39;00m\u001b[38;5;124m greater\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    373\u001b[0m             \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m than the number of samples: n_samples=\u001b[39m\u001b[38;5;132;01m{1}\u001b[39;00m\u001b[38;5;124m.\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    374\u001b[0m         )\u001b[38;5;241m.\u001b[39mformat(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mn_splits, n_samples)\n\u001b[0;32m    375\u001b[0m     )\n\u001b[0;32m    377\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m train, test \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28msuper\u001b[39m()\u001b[38;5;241m.\u001b[39msplit(X, y, groups):\n\u001b[0;32m    378\u001b[0m     \u001b[38;5;28;01myield\u001b[39;00m train, test\n",
      "\u001b[1;31mValueError\u001b[0m: Cannot have number of splits n_splits=5 greater than the number of samples: n_samples=3."
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import RandomizedSearchCV\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "\n",
    "\n",
    "X = df.drop('target', axis=1) \n",
    "y = df['target']\n",
    "\n",
    "# Define hyperparameter grid\n",
    "rf_grid = {\n",
    "    'n_estimators': [100, 200, 500],\n",
    "    'max_depth': [5, 8, 15],\n",
    "    'min_samples_split': [2, 5, 10]\n",
    "}\n",
    "\n",
    "# Set up RandomizedSearchCV \n",
    "rf_search = RandomizedSearchCV(\n",
    "    RandomForestClassifier(),\n",
    "    rf_grid,\n",
    "    n_jobs=-1,\n",
    "    cv=5,\n",
    "    scoring='accuracy'\n",
    ")\n",
    "\n",
    "# Fit on data\n",
    "rf_search.fit(X, y)\n",
    "\n",
    "# Get best hyperparameters \n",
    "print(rf_search.best_params_)\n",
    "\n",
    "# Refit on whole dataset\n",
    "rf_best = RandomForestClassifier(**rf_search.best_params_)\n",
    "rf_best.fit(X, y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5b73e2be",
   "metadata": {},
   "outputs": [],
   "source": [
    "ACCURACY"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "d9425fc0",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "ff317e96",
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "pos_label=1 is not a valid label. It should be one of [7, 8]",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[36], line 8\u001b[0m\n\u001b[0;32m      6\u001b[0m \u001b[38;5;66;03m# Calculate metrics\u001b[39;00m\n\u001b[0;32m      7\u001b[0m accuracy \u001b[38;5;241m=\u001b[39m accuracy_score(y_test, y_pred)\n\u001b[1;32m----> 8\u001b[0m precision \u001b[38;5;241m=\u001b[39m precision_score(y_test, y_pred)\n\u001b[0;32m      9\u001b[0m recall \u001b[38;5;241m=\u001b[39m recall_score(y_test, y_pred)\n\u001b[0;32m     10\u001b[0m f1 \u001b[38;5;241m=\u001b[39m f1_score(y_test, y_pred)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\sklearn\\utils\\_param_validation.py:211\u001b[0m, in \u001b[0;36mvalidate_params.<locals>.decorator.<locals>.wrapper\u001b[1;34m(*args, **kwargs)\u001b[0m\n\u001b[0;32m    205\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[0;32m    206\u001b[0m     \u001b[38;5;28;01mwith\u001b[39;00m config_context(\n\u001b[0;32m    207\u001b[0m         skip_parameter_validation\u001b[38;5;241m=\u001b[39m(\n\u001b[0;32m    208\u001b[0m             prefer_skip_nested_validation \u001b[38;5;129;01mor\u001b[39;00m global_skip_validation\n\u001b[0;32m    209\u001b[0m         )\n\u001b[0;32m    210\u001b[0m     ):\n\u001b[1;32m--> 211\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m func(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n\u001b[0;32m    212\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m InvalidParameterError \u001b[38;5;28;01mas\u001b[39;00m e:\n\u001b[0;32m    213\u001b[0m     \u001b[38;5;66;03m# When the function is just a wrapper around an estimator, we allow\u001b[39;00m\n\u001b[0;32m    214\u001b[0m     \u001b[38;5;66;03m# the function to delegate validation to the estimator, but we replace\u001b[39;00m\n\u001b[0;32m    215\u001b[0m     \u001b[38;5;66;03m# the name of the estimator by the name of the function in the error\u001b[39;00m\n\u001b[0;32m    216\u001b[0m     \u001b[38;5;66;03m# message to avoid confusion.\u001b[39;00m\n\u001b[0;32m    217\u001b[0m     msg \u001b[38;5;241m=\u001b[39m re\u001b[38;5;241m.\u001b[39msub(\n\u001b[0;32m    218\u001b[0m         \u001b[38;5;124mr\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mparameter of \u001b[39m\u001b[38;5;124m\\\u001b[39m\u001b[38;5;124mw+ must be\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[0;32m    219\u001b[0m         \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mparameter of \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mfunc\u001b[38;5;241m.\u001b[39m\u001b[38;5;18m__qualname__\u001b[39m\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m must be\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[0;32m    220\u001b[0m         \u001b[38;5;28mstr\u001b[39m(e),\n\u001b[0;32m    221\u001b[0m     )\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\sklearn\\metrics\\_classification.py:2127\u001b[0m, in \u001b[0;36mprecision_score\u001b[1;34m(y_true, y_pred, labels, pos_label, average, sample_weight, zero_division)\u001b[0m\n\u001b[0;32m   1970\u001b[0m \u001b[38;5;129m@validate_params\u001b[39m(\n\u001b[0;32m   1971\u001b[0m     {\n\u001b[0;32m   1972\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124my_true\u001b[39m\u001b[38;5;124m\"\u001b[39m: [\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124marray-like\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124msparse matrix\u001b[39m\u001b[38;5;124m\"\u001b[39m],\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m   1996\u001b[0m     zero_division\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mwarn\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[0;32m   1997\u001b[0m ):\n\u001b[0;32m   1998\u001b[0m \u001b[38;5;250m    \u001b[39m\u001b[38;5;124;03m\"\"\"Compute the precision.\u001b[39;00m\n\u001b[0;32m   1999\u001b[0m \n\u001b[0;32m   2000\u001b[0m \u001b[38;5;124;03m    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of\u001b[39;00m\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m   2125\u001b[0m \u001b[38;5;124;03m    array([0.5, 1. , 1. ])\u001b[39;00m\n\u001b[0;32m   2126\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n\u001b[1;32m-> 2127\u001b[0m     p, _, _, _ \u001b[38;5;241m=\u001b[39m precision_recall_fscore_support(\n\u001b[0;32m   2128\u001b[0m         y_true,\n\u001b[0;32m   2129\u001b[0m         y_pred,\n\u001b[0;32m   2130\u001b[0m         labels\u001b[38;5;241m=\u001b[39mlabels,\n\u001b[0;32m   2131\u001b[0m         pos_label\u001b[38;5;241m=\u001b[39mpos_label,\n\u001b[0;32m   2132\u001b[0m         average\u001b[38;5;241m=\u001b[39maverage,\n\u001b[0;32m   2133\u001b[0m         warn_for\u001b[38;5;241m=\u001b[39m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mprecision\u001b[39m\u001b[38;5;124m\"\u001b[39m,),\n\u001b[0;32m   2134\u001b[0m         sample_weight\u001b[38;5;241m=\u001b[39msample_weight,\n\u001b[0;32m   2135\u001b[0m         zero_division\u001b[38;5;241m=\u001b[39mzero_division,\n\u001b[0;32m   2136\u001b[0m     )\n\u001b[0;32m   2137\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m p\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\sklearn\\utils\\_param_validation.py:184\u001b[0m, in \u001b[0;36mvalidate_params.<locals>.decorator.<locals>.wrapper\u001b[1;34m(*args, **kwargs)\u001b[0m\n\u001b[0;32m    182\u001b[0m global_skip_validation \u001b[38;5;241m=\u001b[39m get_config()[\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mskip_parameter_validation\u001b[39m\u001b[38;5;124m\"\u001b[39m]\n\u001b[0;32m    183\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m global_skip_validation:\n\u001b[1;32m--> 184\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m func(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n\u001b[0;32m    186\u001b[0m func_sig \u001b[38;5;241m=\u001b[39m signature(func)\n\u001b[0;32m    188\u001b[0m \u001b[38;5;66;03m# Map *args/**kwargs to the function signature\u001b[39;00m\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\sklearn\\metrics\\_classification.py:1721\u001b[0m, in \u001b[0;36mprecision_recall_fscore_support\u001b[1;34m(y_true, y_pred, beta, labels, pos_label, average, warn_for, sample_weight, zero_division)\u001b[0m\n\u001b[0;32m   1563\u001b[0m \u001b[38;5;250m\u001b[39m\u001b[38;5;124;03m\"\"\"Compute precision, recall, F-measure and support for each class.\u001b[39;00m\n\u001b[0;32m   1564\u001b[0m \n\u001b[0;32m   1565\u001b[0m \u001b[38;5;124;03mThe precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of\u001b[39;00m\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m   1718\u001b[0m \u001b[38;5;124;03m array([2, 2, 2]))\u001b[39;00m\n\u001b[0;32m   1719\u001b[0m \u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[0;32m   1720\u001b[0m zero_division_value \u001b[38;5;241m=\u001b[39m _check_zero_division(zero_division)\n\u001b[1;32m-> 1721\u001b[0m labels \u001b[38;5;241m=\u001b[39m _check_set_wise_labels(y_true, y_pred, average, labels, pos_label)\n\u001b[0;32m   1723\u001b[0m \u001b[38;5;66;03m# Calculate tp_sum, pred_sum, true_sum ###\u001b[39;00m\n\u001b[0;32m   1724\u001b[0m samplewise \u001b[38;5;241m=\u001b[39m average \u001b[38;5;241m==\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124msamples\u001b[39m\u001b[38;5;124m\"\u001b[39m\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\sklearn\\metrics\\_classification.py:1507\u001b[0m, in \u001b[0;36m_check_set_wise_labels\u001b[1;34m(y_true, y_pred, average, labels, pos_label)\u001b[0m\n\u001b[0;32m   1505\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m pos_label \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;129;01min\u001b[39;00m present_labels:\n\u001b[0;32m   1506\u001b[0m         \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mlen\u001b[39m(present_labels) \u001b[38;5;241m>\u001b[39m\u001b[38;5;241m=\u001b[39m \u001b[38;5;241m2\u001b[39m:\n\u001b[1;32m-> 1507\u001b[0m             \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\n\u001b[0;32m   1508\u001b[0m                 \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mpos_label=\u001b[39m\u001b[38;5;132;01m{\u001b[39;00mpos_label\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m is not a valid label. It \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m   1509\u001b[0m                 \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mshould be one of \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mpresent_labels\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m   1510\u001b[0m             )\n\u001b[0;32m   1511\u001b[0m     labels \u001b[38;5;241m=\u001b[39m [pos_label]\n\u001b[0;32m   1512\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n",
      "\u001b[1;31mValueError\u001b[0m: pos_label=1 is not a valid label. It should be one of [7, 8]"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score\n",
    "\n",
    "# Make predictions on test set\n",
    "y_pred = model.predict(X_test) \n",
    "\n",
    "# Calculate metrics\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "precision = precision_score(y_test, y_pred)\n",
    "recall = recall_score(y_test, y_pred)\n",
    "f1 = f1_score(y_test, y_pred)\n",
    "\n",
    "# Print metrics\n",
    "print(\"Accuracy:\", accuracy)\n",
    "print(\"Precision:\", precision)  \n",
    "print(\"Recall:\", recall)\n",
    "print(\"F1 Score:\", f1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "e708e94f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 96.75164723155889\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import accuracy_score\n",
    "\n",
    "# Make predictions on test set\n",
    "y_pred = model.predict(X_test)\n",
    "\n",
    "# Calculate accuracy \n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "\n",
    "# Calculate accuracy\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(\"Accuracy:\", accuracy)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8815d620",
   "metadata": {},
   "outputs": [],
   "source": [
    "CONFUSION MAERICS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0ad898d0",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import confusion_matrix\n",
    "\n",
    "\n",
    "X = df.drop('target', axis=1)\n",
    "y = df['target']\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "rf = RandomForestClassifier()\n",
    "rf.fit(X_train, y_train)\n",
    "\n",
    "y_pred = rf.predict(X_test)\n",
    "\n",
    "confusion = confusion_matrix(y_test, y_pred)\n",
    "print(confusion)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "00393db0",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import confusion_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "f56f9a03",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0 1]\n",
      " [0 0]]\n"
     ]
    }
   ],
   "source": [
    "X = df.drop('target', axis=1)\n",
    "y = df['target']\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "rf = RandomForestClassifier()\n",
    "rf.fit(X_train, y_train)\n",
    "\n",
    "y_pred = rf.predict(X_test)\n",
    "\n",
    "confusion = confusion_matrix(y_test, y_pred)\n",
    "print(confusion)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "cd224694",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion Matrix:\n",
      "\n",
      " [[5 0]\n",
      " [2 3]]\n",
      "\n",
      "Precision: 1.0\n",
      "\n",
      "Recall: 0.6\n",
      "\n",
      "F1 Score: 0.7499999999999999\n",
      "\n",
      "False Negative rate: 0.4\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score\n",
    "\n",
    "# Example predictions and true labels\n",
    "y_true = [1, 1, 0, 1, 0, 0, 1, 0, 1, 0]  \n",
    "y_pred = [1, 0, 0, 1, 0, 0, 1, 0, 0, 0]\n",
    "\n",
    "# Calculate confusion matrix\n",
    "cnf_matrix = confusion_matrix(y_true, y_pred)\n",
    "\n",
    "# Calculate metrics from confusion matrix\n",
    "tp = cnf_matrix[1,1]\n",
    "fp = cnf_matrix[0,1]\n",
    "fn = cnf_matrix[1,0]\n",
    "tn = cnf_matrix[0,0]\n",
    "\n",
    "precision = tp / (tp + fp)\n",
    "recall = tp / (tp + fn) \n",
    "f1 = 2 * (precision * recall) / (precision + recall)\n",
    "fnr = fn / (fn + tp)\n",
    "\n",
    "# Print metrics\n",
    "print('Confusion Matrix:\\n\\n', cnf_matrix)\n",
    "print('\\nPrecision:', precision)\n",
    "print('\\nRecall:', recall)\n",
    "print('\\nF1 Score:', f1) \n",
    "print('\\nFalse Negative rate:', fnr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4ae6affe",
   "metadata": {},
   "outputs": [],
   "source": [
    "CONCLUSION"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7fd88ca5",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "- Most patients presented with cough and/or fever symptoms, which are common indicators of respiratory illness. Only 1 patient\n",
    "had no symptoms at all.\n",
    "\n",
    "- 5 out of 10 patients reported known contact with other infected individuals. This suggests potential human-to-human\n",
    "transmission of the illness.\n",
    "\n",
    "- Shortness of breath and sore throat were less commonly reported compared to cough and fever. This may indicate the main\n",
    "symptoms to screen for.\n",
    "\n",
    "- Nearly all patients reported headache along with other symptoms. Headache seems to regularly accompany the primary symptoms\n",
    "like cough and fever.\n",
    "\n",
    "- There does not appear to be an obvious correlation between specific symptoms and known contact with infected individuals\n",
    "in this small sample. Larger samples would be needed to draw conclusions. \n",
    "\n",
    "- The symptom profile is consistent with a respiratory viral illness given cough, fever, and shortness of breath are commonly \n",
    "reported. Additional lab testing would be required to confirm the exact etiology.\n",
    "\n",
    "- Patients with a cough, fever, sore throat and known contact with infected individuals may be highest priority for testing\n",
    "and isolation to prevent further transmission.\n",
    "\n",
    "In summary, these data are indicative of a respiratory illness spreading via human contact, with cough, fever, and headache \n",
    "as predominant symptoms. More data and lab testing would be needed to confirm the exact cause and patterns. Let me know if this\n",
    "provides a helpful high-level conclusion based on the information given!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "86895c5a",
   "metadata": {},
   "outputs": [],
   "source": [
    "OVERALL METHODS USED IN THIS PROJECT.\n",
    "\n",
    "Method 1:\n",
    "- Uses RandomizedSearchCV to tune hyperparameters of RandomForestClassifier\n",
    "- Evaluates using 5-fold cross validation and accuracy scoring\n",
    "- Prints best hyperparameters and refits model on full dataset \n",
    "\n",
    "Pros:\n",
    "- Implements hyperparameter tuning rather than default hyperparameters\n",
    "- Uses cross validation to evaluate robustly \n",
    "- Refits on full dataset to create best model before final evaluation\n",
    "\n",
    "Cons: \n",
    "- Only evaluates based on accuracy, which has limitations if classes are imbalanced\n",
    "\n",
    "Method 2:\n",
    "- Calculates accuracy, precision, recall and F1 score\n",
    "- Uses classification metrics like precision and recall that account for imbalances\n",
    "\n",
    "Pros:\n",
    "- Evaluates on multiple metrics beyond just accuracy\n",
    "- Calculates precision and recall which reveal insights into types of errors\n",
    "\n",
    "Cons:\n",
    "- Does not clearly implement training/validation split, cross validation, or hyperparameter tuning\n",
    "\n",
    "Method 3:\n",
    "- Generates confusion matrix and derives metrics from it\n",
    "- Calculates precision, recall, F1, and false negative rate\n",
    "\n",
    "Pros: \n",
    "- Confusion matrix gives complete picture of model performance\n",
    "- Computes advanced classification metrics from confusion matrix\n",
    "\n",
    "Cons:\n",
    "- Does not implement model training process\n",
    "\n",
    "Overall, I would recommend Method 1 as the best approach as it utilizes cross validation, hyperparameter tuning, and evaluation\n",
    "on a holdout test set. Method 2 and 3 compute important metrics, but lack the model development process. \n",
    "\n",
    "The ideal approach would be to utilize aspects of all three:\n",
    "- Hyperparameter tuning and cross validation from Method 1  \n",
    "- Evaluation metrics from Method 2 and Method 3\n",
    "- Refitting on full dataset before final test evaluation\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ed0c97e2",
   "metadata": {},
   "outputs": [],
   "source": [
    "THESE ARE SOME OF THE BETTER APPORCHES.\n",
    "Method 1 implements hyperparameter tuning through RandomizedSearchCV and evaluates using cross-validation on a metric like \n",
    "accuracy. However, accuracy alone can be misleading if classes are imbalanced.\n",
    "\n",
    "Method 2 calculates additional evaluation metrics like precision, recall and F1 score that provide more insights into model \n",
    "performance, especially for imbalanced classes. However, it does not clearly implement cross-validation or hyperparameter tuninG\n",
    ".\n",
    "\n",
    "Method 3 generates the confusion matrix and derives precision, recall, F1 and false negative rate from it. This gives a very \n",
    "comprehensive view of model performance. But it does not cover the model training process.\n",
    "\n",
    "Given these factors, I would recommend a combination of Method 1 and Method 3:\n",
    "\n",
    "- Use RandomizedSearchCV for hyperparameter tuning with cross-validation as in Method 1\n",
    "- Evaluate on metrics like precision, recall, F1 score, confusion matrix as in Method 3\n",
    "- Refit the best model on the full dataset as in Method 1\n",
    "\n",
    "This combines the benefits of proper model training and tuning from Method 1 with the comprehensive and insightful evaluation\n",
    "metrics from Method 3. \n",
    "\n",
    "The hyperparameter tuning and cross-validation of Method 1 helps prevent overfitting and find the optimal model parameters.\n",
    "\n",
    "The evaluation metrics of Method 3 give a complete view of performance, especially for imbalanced classes where accuracy has\n",
    "limitations.\n",
    "\n"
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
