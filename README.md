# Social Distance Detector

Team Name:CTRL_WE

Other Contributors:srividhya_14, koushika3


During the covid 19 pandemic the main priority is the following of covid19 protocols. Social Distancing of 6 feet must be strictly followed between any 2 person. But it has been observed that People are not following social distancing(>6 feet) and are moving in clusters.

To ensure social distancing is maintained strictly , our project monitors the people walking on the streets with the help of CCTV camera / second hand mobile phone cameras.

Developed using :Python , OpenCV , YoloV3


# Functionalities of our project:

1.Find the distance between 2 nearest person and classify them as safe (green boxed) or unsafe(red boxed) <br/>


2.Give alarm if cluster of people is identified prompting everyone to follow social distancing. <br/>


3.Send SMS alert to concerned authorities if the risk factor is found high. <br/>

--> social_distance_detector.py

    In this program, we use the captured video stream to identify whether the people are following social distancing norms. Green box is drawn outlining the person if they are within distance of minimum 6 feet from others, else Red box is drawn indicating they are not maintaing a safe distnace from others.

--> social_distance_with_storage.py

    It has all the functionality as the above program, additionally some extra features:

    1. We have added a parameter called Risk Factor(RF = #people within unsafe distance/ #people) 

    2. We store Timestamp and RF every second in a CSV file, which can be used for further analysis.

--> with_bolt_mysql_sms.py

    Furthermore, in this program we have integrated Bolt IoT Wifi module to give out an alarm whenever the RF > 0.3. This alarm will alert the people to maintain the social diatance. Once RF <= 0.3 the alarms stops ringing. Also, if the RF > 0.5 concerned authorites will be alerted via SMS facility in our program. In Addition to that, we are storing RF and Timestamp in MySQL database in this program instead of a CSV file.


Attached the ppt with detailed workflow and description. <br/>


https://docs.google.com/presentation/d/1oqNJy7FIPf_ngDeQmhDqBee6VBjTzPcphjc_Pv8-8do/edit?usp=sharing

[Developed during SRM Hacktrix Hackathon(April2-April4)]
