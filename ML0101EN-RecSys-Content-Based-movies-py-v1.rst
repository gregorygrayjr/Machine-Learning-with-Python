
.. raw:: html

   <h1 align="center">

CONTENT-BASED FILTERING

.. raw:: html

   </h1>

Recommendation systems are a collection of algorithms used to recommend
items to users based on information taken from the user. These systems
have become ubiquitous, and can be commonly seen in online stores,
movies databases and job finders. In this notebook, we will explore
Content-based recommendation systems and implement a simple version of
one using Python and the Pandas library.

Table of contents
~~~~~~~~~~~~~~~~~

.. raw:: html

   <div class="alert alert-block alert-info" style="margin-top: 20px">

::

    <ul>
        <li>Acquiring the Data</li>
        <li>Preprocessing</li>
        <li>Content-Based Filtering</li>
    </ul>

.. raw:: html

   </div>

 # Acquiring the Data

| To acquire and extract the data, simply run the following Bash
  scripts:
| Dataset acquired from
  `GroupLens <http://grouplens.org/datasets/movielens/>`__. Lets
  download the dataset. To download the data, we will use **``!wget``**
  to download it from IBM Object Storage.
| **Did you know?** When it comes to Machine Learning, you will likely
  be working with large datasets. As a business, where can you host your
  data? IBM is offering a unique opportunity for businesses, with 10 Tb
  of IBM Cloud Object Storage: `Sign up now for
  free <http://cocl.us/ML0101EN-IBM-Offer-CC>`__

.. code:: ipython3

    !wget -O moviedataset.zip https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/moviedataset.zip
    print('unziping ...')
    !unzip -o -j moviedataset.zip 


.. parsed-literal::

    --2019-03-06 12:37:07--  https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/moviedataset.zip
    Resolving s3-api.us-geo.objectstorage.softlayer.net (s3-api.us-geo.objectstorage.softlayer.net)... 67.228.254.193
    Connecting to s3-api.us-geo.objectstorage.softlayer.net (s3-api.us-geo.objectstorage.softlayer.net)|67.228.254.193|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 160301210 (153M) [application/zip]
    Saving to: ‘moviedataset.zip’
    
    moviedataset.zip    100%[=====================>] 152.88M  20.2MB/s   in 6.1s   
    
    2019-03-06 12:37:13 (25.0 MB/s) - ‘moviedataset.zip’ saved [160301210/160301210]
    
    unziping ...
    Archive:  moviedataset.zip
      inflating: links.csv               
      inflating: movies.csv              
      inflating: ratings.csv             
      inflating: README.txt              
      inflating: tags.csv                


Now you're ready to start working with the data!

 # Preprocessing

First, let's get all of the imports out of the way:

.. code:: ipython3

    #Dataframe manipulation library
    import pandas as pd
    #Math functions, we'll only need the sqrt function so let's import only that
    from math import sqrt
    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline

Now let's read each file into their Dataframes:

.. code:: ipython3

    #Storing the movie information into a pandas dataframe
    movies_df = pd.read_csv('movies.csv')
    #Storing the user information into a pandas dataframe
    ratings_df = pd.read_csv('ratings.csv')
    #Head is a function that gets the first N rows of a dataframe. N's default is 5.
    movies_df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>movieId</th>
          <th>title</th>
          <th>genres</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>Toy Story (1995)</td>
          <td>Adventure|Animation|Children|Comedy|Fantasy</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2</td>
          <td>Jumanji (1995)</td>
          <td>Adventure|Children|Fantasy</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3</td>
          <td>Grumpier Old Men (1995)</td>
          <td>Comedy|Romance</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4</td>
          <td>Waiting to Exhale (1995)</td>
          <td>Comedy|Drama|Romance</td>
        </tr>
        <tr>
          <th>4</th>
          <td>5</td>
          <td>Father of the Bride Part II (1995)</td>
          <td>Comedy</td>
        </tr>
      </tbody>
    </table>
    </div>



Let's also remove the year from the **title** column by using pandas'
replace function and store in a new **year** column.

.. code:: ipython3

    #Using regular expressions to find a year stored between parentheses
    #We specify the parantheses so we don't conflict with movies that have years in their titles
    movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
    #Removing the parentheses
    movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
    #Removing the years from the 'title' column
    movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
    #Applying the strip function to get rid of any ending whitespace characters that may have appeared
    movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
    movies_df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>movieId</th>
          <th>title</th>
          <th>genres</th>
          <th>year</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>Toy Story</td>
          <td>Adventure|Animation|Children|Comedy|Fantasy</td>
          <td>1995</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2</td>
          <td>Jumanji</td>
          <td>Adventure|Children|Fantasy</td>
          <td>1995</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3</td>
          <td>Grumpier Old Men</td>
          <td>Comedy|Romance</td>
          <td>1995</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4</td>
          <td>Waiting to Exhale</td>
          <td>Comedy|Drama|Romance</td>
          <td>1995</td>
        </tr>
        <tr>
          <th>4</th>
          <td>5</td>
          <td>Father of the Bride Part II</td>
          <td>Comedy</td>
          <td>1995</td>
        </tr>
      </tbody>
    </table>
    </div>



With that, let's also split the values in the **Genres** column into a
**list of Genres** to simplify future use. This can be achieved by
applying Python's split string function on the correct column.

.. code:: ipython3

    #Every genre is separated by a | so we simply have to call the split function on |
    movies_df['genres'] = movies_df.genres.str.split('|')
    movies_df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>movieId</th>
          <th>title</th>
          <th>genres</th>
          <th>year</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>Toy Story</td>
          <td>[Adventure, Animation, Children, Comedy, Fantasy]</td>
          <td>1995</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2</td>
          <td>Jumanji</td>
          <td>[Adventure, Children, Fantasy]</td>
          <td>1995</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3</td>
          <td>Grumpier Old Men</td>
          <td>[Comedy, Romance]</td>
          <td>1995</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4</td>
          <td>Waiting to Exhale</td>
          <td>[Comedy, Drama, Romance]</td>
          <td>1995</td>
        </tr>
        <tr>
          <th>4</th>
          <td>5</td>
          <td>Father of the Bride Part II</td>
          <td>[Comedy]</td>
          <td>1995</td>
        </tr>
      </tbody>
    </table>
    </div>



Since keeping genres in a list format isn't optimal for the
content-based recommendation system technique, we will use the One Hot
Encoding technique to convert the list of genres to a vector where each
column corresponds to one possible value of the feature. This encoding
is needed for feeding categorical data. In this case, we store every
different genre in columns that contain either 1 or 0. 1 shows that a
movie has that genre and 0 shows that it doesn't. Let's also store this
dataframe in another variable since genres won't be important for our
first recommendation system.

.. code:: ipython3

    #Copying the movie dataframe into a new one since we won't need to use the genre information in our first case.
    moviesWithGenres_df = movies_df.copy()
    
    #For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column
    for index, row in movies_df.iterrows():
        for genre in row['genres']:
            moviesWithGenres_df.at[index, genre] = 1
    #Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
    moviesWithGenres_df = moviesWithGenres_df.fillna(0)
    moviesWithGenres_df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>movieId</th>
          <th>title</th>
          <th>genres</th>
          <th>year</th>
          <th>Adventure</th>
          <th>Animation</th>
          <th>Children</th>
          <th>Comedy</th>
          <th>Fantasy</th>
          <th>Romance</th>
          <th>...</th>
          <th>Horror</th>
          <th>Mystery</th>
          <th>Sci-Fi</th>
          <th>IMAX</th>
          <th>Documentary</th>
          <th>War</th>
          <th>Musical</th>
          <th>Western</th>
          <th>Film-Noir</th>
          <th>(no genres listed)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>Toy Story</td>
          <td>[Adventure, Animation, Children, Comedy, Fantasy]</td>
          <td>1995</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>...</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2</td>
          <td>Jumanji</td>
          <td>[Adventure, Children, Fantasy]</td>
          <td>1995</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>...</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3</td>
          <td>Grumpier Old Men</td>
          <td>[Comedy, Romance]</td>
          <td>1995</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>...</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4</td>
          <td>Waiting to Exhale</td>
          <td>[Comedy, Drama, Romance]</td>
          <td>1995</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>...</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>4</th>
          <td>5</td>
          <td>Father of the Bride Part II</td>
          <td>[Comedy]</td>
          <td>1995</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>...</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
        </tr>
      </tbody>
    </table>
    <p>5 rows × 24 columns</p>
    </div>



Next, let's look at the ratings dataframe.

.. code:: ipython3

    ratings_df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>userId</th>
          <th>movieId</th>
          <th>rating</th>
          <th>timestamp</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>169</td>
          <td>2.5</td>
          <td>1204927694</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1</td>
          <td>2471</td>
          <td>3.0</td>
          <td>1204927438</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1</td>
          <td>48516</td>
          <td>5.0</td>
          <td>1204927435</td>
        </tr>
        <tr>
          <th>3</th>
          <td>2</td>
          <td>2571</td>
          <td>3.5</td>
          <td>1436165433</td>
        </tr>
        <tr>
          <th>4</th>
          <td>2</td>
          <td>109487</td>
          <td>4.0</td>
          <td>1436165496</td>
        </tr>
      </tbody>
    </table>
    </div>



Every row in the ratings dataframe has a user id associated with at
least one movie, a rating and a timestamp showing when they reviewed it.
We won't be needing the timestamp column, so let's drop it to save on
memory.

.. code:: ipython3

    #Drop removes a specified row or column from a dataframe
    ratings_df = ratings_df.drop('timestamp', 1)
    ratings_df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>userId</th>
          <th>movieId</th>
          <th>rating</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>169</td>
          <td>2.5</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1</td>
          <td>2471</td>
          <td>3.0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1</td>
          <td>48516</td>
          <td>5.0</td>
        </tr>
        <tr>
          <th>3</th>
          <td>2</td>
          <td>2571</td>
          <td>3.5</td>
        </tr>
        <tr>
          <th>4</th>
          <td>2</td>
          <td>109487</td>
          <td>4.0</td>
        </tr>
      </tbody>
    </table>
    </div>



 # Content-Based recommendation system

Now, let's take a look at how to implement **Content-Based** or
**Item-Item recommendation systems**. This technique attempts to figure
out what a user's favourite aspects of an item is, and then recommends
items that present those aspects. In our case, we're going to try to
figure out the input's favorite genres from the movies and ratings
given.

Let's begin by creating an input user to recommend movies to:

Notice: To add more movies, simply increase the amount of elements in
the **userInput**. Feel free to add more in! Just be sure to write it in
with capital letters and if a movie starts with a "The", like "The
Matrix" then write it in like this: 'Matrix, The' .

.. code:: ipython3

    userInput = [
                {'title':'Breakfast Club, The', 'rating':5},
                {'title':'Toy Story', 'rating':3.5},
                {'title':'Jumanji', 'rating':2},
                {'title':"Pulp Fiction", 'rating':5},
                {'title':'Akira', 'rating':4.5}
             ] 
    inputMovies = pd.DataFrame(userInput)
    inputMovies




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>rating</th>
          <th>title</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>5.0</td>
          <td>Breakfast Club, The</td>
        </tr>
        <tr>
          <th>1</th>
          <td>3.5</td>
          <td>Toy Story</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2.0</td>
          <td>Jumanji</td>
        </tr>
        <tr>
          <th>3</th>
          <td>5.0</td>
          <td>Pulp Fiction</td>
        </tr>
        <tr>
          <th>4</th>
          <td>4.5</td>
          <td>Akira</td>
        </tr>
      </tbody>
    </table>
    </div>



Add movieId to input user
^^^^^^^^^^^^^^^^^^^^^^^^^

With the input complete, let's extract the input movie's ID's from the
movies dataframe and add them into it.

We can achieve this by first filtering out the rows that contain the
input movie's title and then merging this subset with the input
dataframe. We also drop unnecessary columns for the input to save memory
space.

.. code:: ipython3

    #Filtering out the movies by title
    inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
    #Then merging it so we can get the movieId. It's implicitly merging it by title.
    inputMovies = pd.merge(inputId, inputMovies)
    #Dropping information we won't use from the input dataframe
    inputMovies = inputMovies.drop('genres', 1).drop('year', 1)
    #Final input dataframe
    #If a movie you added in above isn't here, then it might not be in the original 
    #dataframe or it might spelled differently, please check capitalisation.
    inputMovies




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>movieId</th>
          <th>title</th>
          <th>rating</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>Toy Story</td>
          <td>3.5</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2</td>
          <td>Jumanji</td>
          <td>2.0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>296</td>
          <td>Pulp Fiction</td>
          <td>5.0</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1274</td>
          <td>Akira</td>
          <td>4.5</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1968</td>
          <td>Breakfast Club, The</td>
          <td>5.0</td>
        </tr>
      </tbody>
    </table>
    </div>



We're going to start by learning the input's preferences, so let's get
the subset of movies that the input has watched from the Dataframe
containing genres defined with binary values.

.. code:: ipython3

    #Filtering out the movies from the input
    userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]
    userMovies




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>movieId</th>
          <th>title</th>
          <th>genres</th>
          <th>year</th>
          <th>Adventure</th>
          <th>Animation</th>
          <th>Children</th>
          <th>Comedy</th>
          <th>Fantasy</th>
          <th>Romance</th>
          <th>...</th>
          <th>Horror</th>
          <th>Mystery</th>
          <th>Sci-Fi</th>
          <th>IMAX</th>
          <th>Documentary</th>
          <th>War</th>
          <th>Musical</th>
          <th>Western</th>
          <th>Film-Noir</th>
          <th>(no genres listed)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>Toy Story</td>
          <td>[Adventure, Animation, Children, Comedy, Fantasy]</td>
          <td>1995</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>...</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2</td>
          <td>Jumanji</td>
          <td>[Adventure, Children, Fantasy]</td>
          <td>1995</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>...</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>293</th>
          <td>296</td>
          <td>Pulp Fiction</td>
          <td>[Comedy, Crime, Drama, Thriller]</td>
          <td>1994</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>...</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>1246</th>
          <td>1274</td>
          <td>Akira</td>
          <td>[Action, Adventure, Animation, Sci-Fi]</td>
          <td>1988</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>...</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>1885</th>
          <td>1968</td>
          <td>Breakfast Club, The</td>
          <td>[Comedy, Drama]</td>
          <td>1985</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>...</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
        </tr>
      </tbody>
    </table>
    <p>5 rows × 24 columns</p>
    </div>



We'll only need the actual genre table, so let's clean this up a bit by
resetting the index and dropping the movieId, title, genres and year
columns.

.. code:: ipython3

    #Resetting the index to avoid future issues
    userMovies = userMovies.reset_index(drop=True)
    #Dropping unnecessary issues due to save memory and to avoid issues
    userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
    userGenreTable




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Adventure</th>
          <th>Animation</th>
          <th>Children</th>
          <th>Comedy</th>
          <th>Fantasy</th>
          <th>Romance</th>
          <th>Drama</th>
          <th>Action</th>
          <th>Crime</th>
          <th>Thriller</th>
          <th>Horror</th>
          <th>Mystery</th>
          <th>Sci-Fi</th>
          <th>IMAX</th>
          <th>Documentary</th>
          <th>War</th>
          <th>Musical</th>
          <th>Western</th>
          <th>Film-Noir</th>
          <th>(no genres listed)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
        </tr>
      </tbody>
    </table>
    </div>



Now we're ready to start learning the input's preferences!

To do this, we're going to turn each genre into weights. We can do this
by using the input's reviews and multiplying them into the input's genre
table and then summing up the resulting table by column. This operation
is actually a dot product between a matrix and a vector, so we can
simply accomplish by calling Pandas's "dot" function.

.. code:: ipython3

    inputMovies['rating']




.. parsed-literal::

    0    3.5
    1    2.0
    2    5.0
    3    4.5
    4    5.0
    Name: rating, dtype: float64



.. code:: ipython3

    #Dot produt to get weights
    userProfile = userGenreTable.transpose().dot(inputMovies['rating'])
    #The user profile
    userProfile




.. parsed-literal::

    Adventure             10.0
    Animation              8.0
    Children               5.5
    Comedy                13.5
    Fantasy                5.5
    Romance                0.0
    Drama                 10.0
    Action                 4.5
    Crime                  5.0
    Thriller               5.0
    Horror                 0.0
    Mystery                0.0
    Sci-Fi                 4.5
    IMAX                   0.0
    Documentary            0.0
    War                    0.0
    Musical                0.0
    Western                0.0
    Film-Noir              0.0
    (no genres listed)     0.0
    dtype: float64



Now, we have the weights for every of the user's preferences. This is
known as the User Profile. Using this, we can recommend movies that
satisfy the user's preferences.

Let's start by extracting the genre table from the original dataframe:

.. code:: ipython3

    #Now let's get the genres of every movie in our original dataframe
    genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
    #And drop the unnecessary information
    genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
    genreTable.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Adventure</th>
          <th>Animation</th>
          <th>Children</th>
          <th>Comedy</th>
          <th>Fantasy</th>
          <th>Romance</th>
          <th>Drama</th>
          <th>Action</th>
          <th>Crime</th>
          <th>Thriller</th>
          <th>Horror</th>
          <th>Mystery</th>
          <th>Sci-Fi</th>
          <th>IMAX</th>
          <th>Documentary</th>
          <th>War</th>
          <th>Musical</th>
          <th>Western</th>
          <th>Film-Noir</th>
          <th>(no genres listed)</th>
        </tr>
        <tr>
          <th>movieId</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>1</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>5</th>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    genreTable.shape




.. parsed-literal::

    (34208, 20)



With the input's profile and the complete list of movies and their
genres in hand, we're going to take the weighted average of every movie
based on the input profile and recommend the top twenty movies that most
satisfy it.

.. code:: ipython3

    #Multiply the genres by the weights and then take the weighted average
    recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
    recommendationTable_df.head()




.. parsed-literal::

    movieId
    1    0.594406
    2    0.293706
    3    0.188811
    4    0.328671
    5    0.188811
    dtype: float64



.. code:: ipython3

    #Sort our recommendations in descending order
    recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
    #Just a peek at the values
    recommendationTable_df.head()




.. parsed-literal::

    movieId
    5018      0.748252
    26093     0.734266
    27344     0.720280
    148775    0.685315
    6902      0.678322
    dtype: float64



Now here's the recommendation table!

.. code:: ipython3

    #The final recommendation table
    movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())]




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>movieId</th>
          <th>title</th>
          <th>genres</th>
          <th>year</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>664</th>
          <td>673</td>
          <td>Space Jam</td>
          <td>[Adventure, Animation, Children, Comedy, Fanta...</td>
          <td>1996</td>
        </tr>
        <tr>
          <th>1824</th>
          <td>1907</td>
          <td>Mulan</td>
          <td>[Adventure, Animation, Children, Comedy, Drama...</td>
          <td>1998</td>
        </tr>
        <tr>
          <th>2902</th>
          <td>2987</td>
          <td>Who Framed Roger Rabbit?</td>
          <td>[Adventure, Animation, Children, Comedy, Crime...</td>
          <td>1988</td>
        </tr>
        <tr>
          <th>4923</th>
          <td>5018</td>
          <td>Motorama</td>
          <td>[Adventure, Comedy, Crime, Drama, Fantasy, Mys...</td>
          <td>1991</td>
        </tr>
        <tr>
          <th>6793</th>
          <td>6902</td>
          <td>Interstate 60</td>
          <td>[Adventure, Comedy, Drama, Fantasy, Mystery, S...</td>
          <td>2002</td>
        </tr>
        <tr>
          <th>8605</th>
          <td>26093</td>
          <td>Wonderful World of the Brothers Grimm, The</td>
          <td>[Adventure, Animation, Children, Comedy, Drama...</td>
          <td>1962</td>
        </tr>
        <tr>
          <th>8783</th>
          <td>26340</td>
          <td>Twelve Tasks of Asterix, The (Les douze travau...</td>
          <td>[Action, Adventure, Animation, Children, Comed...</td>
          <td>1976</td>
        </tr>
        <tr>
          <th>9296</th>
          <td>27344</td>
          <td>Revolutionary Girl Utena: Adolescence of Utena...</td>
          <td>[Action, Adventure, Animation, Comedy, Drama, ...</td>
          <td>1999</td>
        </tr>
        <tr>
          <th>9825</th>
          <td>32031</td>
          <td>Robots</td>
          <td>[Adventure, Animation, Children, Comedy, Fanta...</td>
          <td>2005</td>
        </tr>
        <tr>
          <th>11716</th>
          <td>51632</td>
          <td>Atlantis: Milo's Return</td>
          <td>[Action, Adventure, Animation, Children, Comed...</td>
          <td>2003</td>
        </tr>
        <tr>
          <th>11751</th>
          <td>51939</td>
          <td>TMNT (Teenage Mutant Ninja Turtles)</td>
          <td>[Action, Adventure, Animation, Children, Comed...</td>
          <td>2007</td>
        </tr>
        <tr>
          <th>13250</th>
          <td>64645</td>
          <td>The Wrecking Crew</td>
          <td>[Action, Adventure, Comedy, Crime, Drama, Thri...</td>
          <td>1968</td>
        </tr>
        <tr>
          <th>16055</th>
          <td>81132</td>
          <td>Rubber</td>
          <td>[Action, Adventure, Comedy, Crime, Drama, Film...</td>
          <td>2010</td>
        </tr>
        <tr>
          <th>18312</th>
          <td>91335</td>
          <td>Gruffalo, The</td>
          <td>[Adventure, Animation, Children, Comedy, Drama]</td>
          <td>2009</td>
        </tr>
        <tr>
          <th>22778</th>
          <td>108540</td>
          <td>Ernest &amp; Célestine (Ernest et Célestine)</td>
          <td>[Adventure, Animation, Children, Comedy, Drama...</td>
          <td>2012</td>
        </tr>
        <tr>
          <th>22881</th>
          <td>108932</td>
          <td>The Lego Movie</td>
          <td>[Action, Adventure, Animation, Children, Comed...</td>
          <td>2014</td>
        </tr>
        <tr>
          <th>25218</th>
          <td>117646</td>
          <td>Dragonheart 2: A New Beginning</td>
          <td>[Action, Adventure, Comedy, Drama, Fantasy, Th...</td>
          <td>2000</td>
        </tr>
        <tr>
          <th>26442</th>
          <td>122787</td>
          <td>The 39 Steps</td>
          <td>[Action, Adventure, Comedy, Crime, Drama, Thri...</td>
          <td>1959</td>
        </tr>
        <tr>
          <th>32854</th>
          <td>146305</td>
          <td>Princes and Princesses</td>
          <td>[Animation, Children, Comedy, Drama, Fantasy, ...</td>
          <td>2000</td>
        </tr>
        <tr>
          <th>33509</th>
          <td>148775</td>
          <td>Wizards of Waverly Place: The Movie</td>
          <td>[Adventure, Children, Comedy, Drama, Fantasy, ...</td>
          <td>2009</td>
        </tr>
      </tbody>
    </table>
    </div>



Advantages and Disadvantages of Content-Based Filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Advantages
''''''''''

-  Learns user's preferences
-  Highly personalized for the user

Disadvantages
'''''''''''''

-  Doesn't take into account what others think of the item, so low
   quality item recommendations might happen
-  Extracting data is not always intuitive
-  Determining what characteristics of the item the user dislikes or
   likes is not always obvious

Want to learn more?
-------------------

IBM SPSS Modeler is a comprehensive analytics platform that has many
machine learning algorithms. It has been designed to bring predictive
intelligence to decisions made by individuals, by groups, by systems –
by your enterprise as a whole. A free trial is available through this
course, available here: `SPSS
Modeler <http://cocl.us/ML0101EN-SPSSModeler>`__.

Also, you can use Watson Studio to run these notebooks faster with
bigger datasets. Watson Studio is IBM's leading cloud solution for data
scientists, built by data scientists. With Jupyter notebooks, RStudio,
Apache Spark and popular libraries pre-packaged in the cloud, Watson
Studio enables data scientists to collaborate on their projects without
having to install anything. Join the fast-growing community of Watson
Studio users today with a free account at `Watson
Studio <https://cocl.us/ML0101EN_DSX>`__

Thanks for completing this lesson!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Notebook created by: Saeed Aghabozorgi, Gabriel Garcez Barros Sousa

.. raw:: html

   <hr>

Copyright © 2018 `Cognitive Class <https://cocl.us/DX0108EN_CC>`__. This
notebook and its source code are released under the terms of the `MIT
License <https://bigdatauniversity.com/mit-license/>`__.​
