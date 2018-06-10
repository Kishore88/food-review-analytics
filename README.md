# Running The Application

* To install all the project dependecies

    		sudo pip install -r requirements.txt

* To start the application server.

    		python run.py

* Upload tagged data csv file by making a POST request to the following URL:

            http://localhost:8080/classifiers/rt_NQR5yVAe/upload/


* To train the sentiment and category classification model make a POST request to the following URL:

            http://localhost:8080/classifiers/rt_NQR5yVAe/train/


* To test the API output make a GET request to the following URL:

		    http://localhost:8080/testing?c_id=rt_NQR5yVAe