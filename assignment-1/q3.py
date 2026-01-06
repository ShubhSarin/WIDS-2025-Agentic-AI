from transformers import pipeline

sentiment_analysis = pipeline(task='sentiment-analysis')

reviews = ["""This is how I want to see a spy/espionage movie in Indian cinema. Excellent world and character building, with outstanding set pieces. Every single actor played their roles to perfection! Askhay Khanna stands out the most. However, as others mentioned, the role of Ranveer was not to stand out in the first place, and the director and actor executed that perfectly!""", """Since the beginning Movie pace and Story hook you that much, you'll forget about popcorn on your lap.

Background music 10/10, Highly Impressive. I was complimenting while watching itself.

Casting 10/10, every actor is fit to the role and each and everyone gives 100% Story 9/10, After a decade we are witnessing such a good story and no disturbance by any love song in switzerland.

Direction 10/10, every single shot is too notch. So impressive

Please watch this film in theatre for better experience in terms of music most. I'm filled with compliments for this film.""", """A spy action movie with a runtime of 3 hrs 34 minutes approx, it's engrossing without a single moment of boredom, thanks to the robust screenplay and casting of talented performers. Director Aditya Dhar has been successful in ensuring that each character leaves an indelible mark on the viewers.

It's not solely a Ranveer Singh show, as the director has ensured each character has a memorable presence. R Madhavan, Arjun Rampal, Akshay Khanna, Sanjay Dutt, and Ranveer Singh deliver performances akin to a north Indian person savoring an authentic south Indian meal, with each bite providing satisfaction. Akshay Khanna, who won hearts with Aurangzeb in Chaava, has again impressed as Rehman Dakait in Dhurandhar. His swag, look, and dialogue delivery are mind-blowing. As the movie progresses, Sanjay Dutt brings a new gear, accelerating the pace with his enjoyable screen presence. R Madhavan and Arjun Rampal have limited screen time but make their presence felt. Sara Arjun delivers a decent performance as the lady love. Ranveer Singh's character development is amazing to watch, and his mass performance will be eagerly anticipated in the second part.

From a technical standpoint, Aditya Dhar's direction is astounding, perfectly capturing the essence of Dhurandhar. Sashwat Sachdev's score and music blend seamlessly with the movie's tone. The placement of old Bollywood classics in certain situations is entertaining and thrilling. Vikash Nowlakha's cinematography is commendable, and Shivkumar V Panicker's editing is sharp.""", """Intelligent writing, peak performances, gritty and realistic action. After such a long time, mainstream bollywood finally has a movie which is truly cinema.

It's no simple task to construct a nearly 4 hour long movie without losing your audience, yet dhurandhar manages to do exactly that. In fact, the runtime serves as a strength, allowing you to fully immerse yourself in such a dense, complicated, and dark world.

I say this will change indian cinema because the film is just so thoughtfully built and so different at the same time. It feels like an experimental take on commercial cinema. There's so much aura farming, but dhar's clever script adds a classy weight to them that just isn't present in other bwood flicks.

There are some truly disturbing scenes here, in particular the scene for the 26/11 attack. The shock, the disgust, the sadness. I don't think a film has made me feel so angry in such a long time. Another scene involving arjun rampal's character is straight out of a horror movie. The 18 rating is definitely justified.

Don't miss this masterpiece.""", """The world of Dhurandhar is as mesmerizing as it is dangerous, with every corner of Karachi rendered in cinematic precision. Ranveer Singh delivers a layered performance that anchors the audience amid escalating threats and moral dilemmas. Director Dhar crafts sequences where strategic planning meets brutal action, keeping viewers perpetually on edge. The supporting cast contributes significantly, fleshing out a realistic criminal ecosystem. Cinematography and editing combine to maintain tension, while the score punctuates key moments. The narrative's blend of action, politics, and human drama makes the film a truly immersive experience."""]

analysis = sentiment_analysis(reviews)

for i in range(0, 5):
    print(f"{reviews[i]} \n\n Predicted Label - {analysis[i]['label']}, Score - {analysis[i]['score']}")
    print("===============================================================================")
    
# Screenshot truncated due to long output