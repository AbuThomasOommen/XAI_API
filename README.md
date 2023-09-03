# XAI_API
An Integrated Framework for Classification of Deep Learning Applications using XAI


An Integrated Framework for Classification of Deep Learning Applications using XAI


Abstract


Complex deep learning models benefit greatly from increased transparency and trust thanks to explainable artificial intelligence (XAI). This project thesis presents an innovative web application that showcases the power of XAI techniques in two real-world use cases: Deepfake detection using Convolutional Neural Networks (CNN) and Tweet text classification with Long Short-Term Memory (LSTM) in Natural Language Processing (NLP).
The first application focuses on addressing the alarming rise of deepfake images and videos that pose serious threats to media authenticity and privacy. Leveraging CNN, the web application enables users to discern between genuine and manipulated content by highlighting regions within the image that significantly influence the model's decision. XAI methods, such as Lime (Local Interpretable Model-agnostic Explanations), are employed to provide interpretable insights into the model's decision-making process.
The second application demonstrates the web application's capability to analyze and classify tweet sentiments with LSTM-based NLP techniques. Users can input tweets, and the LSTM model predicts the sentiment, revealing not only the classification but also the most influential words contributing to the model's decision. This fosters better understanding and trust in the model's predictions, making it valuable for various sentiment analysis applications.
The Flask-based web application serves as a user-friendly interface, empowering users to interact seamlessly with the deep learning models. Additionally, the use of interactive visualizations further enhances the understanding of model predictions, boosting user confidence in the system's reliability.
This project thesis showcases the potential of XAI techniques to empower end-users with comprehensive insights into complex deep learning models. By providing interpretable explanations for model predictions, the web application fosters transparency, accountability, and trust in AI systems, thereby advancing the adoption of AI in real-world applications while ensuring responsible AI practices.















Acknowledgements

I would like to express sincere gratitude to Dr. Krishnendu Guha, School of Computer Science and Information Technology, University College Cork, for his continuous support and guidance throughout the course of this project.































Table of Contents 
List of Tables		v
List of Figures		vi
1	Introduction	1
1.1	Motivation	1
1.2	Problem Specification/ Statement	2
1.3	Objective and Focus Areas	3
1.4	Research Foundation	3
1.5	Approach	3
1.6	Thesis Structure	4
2	Literature Review	5
2.1	Background	5
2.2	Lime Method	7
3	Design, Data and Methodology	9
3.1	Tools and Techniques	9
3.2	Workflow and Design	11
3.3	Data Collection and Description	12
3.4	Methodology	13
4	Implementation	16
4.1	Creating the CNN Model for Image Classification	16
4.2	Creating the LSTM Model for Text Classification	20
4.3	Creating the Flask Server side Application	25
4.4	Creating the Flask Client side Web User Interface	29
5	Results	33
6	Conclusion	42
6.1	Conclusion	42
6.2	Future Work	44
7	Bibliography	46







List of Tables


4.1   Model Selection	18


































List of Figures

             2.1    Lime Workflow	8
3.1	Workflow	11
3.2	CIFake Images	12
3.3	Tweet text wordcloud	13
3.4	Lime Algorithm	14
4.1	CNN Architecture 	16
4.2	CNN Model Summary	18
4.3	CNN Model Building	19
4.4	LSTM Architecture	20
4.5	LSTM Model Summary	22
4.6	LSTM Model Creation	23
4.7	LSTM Model Pipeline	24
4.8	Flask Model Loading	25
4.9	Home route	26
4.10	Predict route	27
4.11	Explain route	28
4.12	HTML Code	30
4.13	Snippets of the CSS Code	31
4.14	Interactions Code	32
4.15	Run Model Function	33
5.1	Model Selection	35
5.2	CNN Model Window	35
5.3	Fake Image Classification and Explanation	36
5.4	Real Image Classification and Explanation	37
5.5	LSTM Model Window	38
5.6	Positive Text Classification and Explanation	39
5.7	Neutral Text Classification and Explanation	40
5.8	Negative Text Classification and Explanation	41

6.1	  Concept Design	45








 
1.Introduction


1.1	Motivation

Artificial intelligence (AI) [11], a ground-breaking idea that was previously reserved for science fiction, is now an essential component of our daily life. The journey of AI began with ambitious dreams of creating intelligent machines that could mimic human cognitive abilities. Over the years, as technology evolved and computational power increased, significant progress has been made in the field, leading to the birth of machine learning [10] and, subsequently, deep learning [5]. Machine learning marked a pivotal shift from rule-based programming to data-driven decision-making. i.e.: algorithms and models were designed to learn from data and improve their performance over time without explicit programming. The ability of machines to learn from vast amounts of data and generalize patterns opened up new possibilities in various domains, ranging from image recognition and natural language processing to healthcare and finance. 

Deep learning [5], an advanced subset of machine learning, emerged as the torchbearer for AI's most remarkable achievements. It is inspired by the structure and functioning of the human brain, introduced neural networks with multiple layers, capable of automatically learning hierarchical representations from raw data. These deep neural networks demonstrated unprecedented success in solving complex tasks, surpassing traditional machine learning techniques in accuracy and efficiency. Deep neural networks have enabled AI to do previously unheard-of feats of complexity in areas like speech and picture recognition, language translation, and game play. These models outperformed humans and elevated AI applications to new levels. Deep learning models advanced over time, frequently consisting of millions of interconnected neurons and intricate layers. Despite its astounding precision, a significant obstacle called the "black box" problem occurred. It became harder to understand how models arrived at their predictions or choices as they grew deeper and more complex. The lack of transparency in AI models gave rise to a pressing need for Explainable AI (XAI).

Explainable AI (XAI) [4] aims to shed light on the decision-making process of complex AI systems. XAI aims to provide human-interpretable explanations for the predictions and decisions made by AI models. By shedding light on how the models arrive at their conclusions, XAI enhances the trustworthiness, fairness, and accountability of AI applications, paving the way for their responsible and ethical use. It bridges the gap between accuracy and interpretability, enabling us to understand why a model classified an image as a particular object or labeled a tweet as positive or negative sentiment. The demand for XAI is not only confined to ensuring model fairness and compliance but also extends to building trust with end-users and stakeholders.

In this era of autonomous systems and AI-driven decision-making, it is important to ensure that AI is not an enigma but a trusted partner. With Explainable AI, we can unlock the potential of deep learning while making AI comprehensible, transparent, and accountable, thereby paving the way for responsible and ethically-guided AI advancements.

1.2	Problem Specification/ Statement

In this thesis, the main objective is to create an application to demonstrate the workings of different deep learning classifiers with their explanations. There are two use cases which the application would facilitate: A CNN [6] model which is used to detect deepfakes in images and classify them as fake or real with the explanations, a LSTM [12] model which can do natural language processing (NLP) [15] on tweet text by classifying it as positive, negative or neutral with explanations of their workings. 
One of the main problems in using a deep learning model for classification is deciding which architecture to use. CNN and LSTM models can be created with infinitely different architectures. This can be done by varying the number of layers in the model, changing the kernel size, changing the number of filters, changing the dropout for layers or changing the pool size in the pooling layer, different batch sizes, different range of epochs or various learning rates. Combination of the changes in the architecture or hyperparameters can produce completely different models. The performance of each model will also differ for the same dataset. Few of them may perform much better than the others. The time taken for training varies for each model. Now with the possibilities of the use of transfer learning, the situation has gotten more complicated. These models can outperform the user created architectures as they can provide higher accuracy even though it is only being trained on the limited available data. However, they may not always perform better than the user created models. So, a carefully designed architecture and optimal use of parameters are required to build this application and to make it work seamlessly. This leads to the rise of following questions:

•	Which CNN or LSTM architecture should I use?

•	How many layers should be included in the model?

•	How to select the optimal hyperparameters?

•	Should we use transfer learning and if so, which architecture to choose?

•	How should we source the data?

•	Is there any data preprocessing required?

•	Should Flask or Django be used to build the application?

•	How to integrate both the use cases into the application?

•	How to visualize the explanations alongside the classification output?


1.3	Objective and Focus Areas

The main focus areas and objective of the thesis are listed below:

•	Procure the data for training the model.

•	Analyze the data.

•	Pre-process the data and make it compatible for the models.

•	Try creating different models and architectures in both use cases.

•	Train the models on training datasets.

•	Evaluate the performance of the models on test datasets.

•	Create an application to demonstrate the capabilities of explainable AI and its working for both the use cases.

•	Integrate both the models to work in the application seamlessly.

•	Visualize the explanations along with the model output.


1.4	Research Foundation

The requirement to comprehend and interpret complicated machine learning models, particularly those with black-box qualities like deep neural networks, gave rise to the research basis for Explainable AI (XAI). Although these models frequently perform admirably on a variety of tasks, their decision-making procedures can be difficult to understand, raising questions about their dependability, fairness, and safety in important applications. The development of XAI techniques is rooted in a combination of fields, including machine learning, computer science, human-computer interaction, and cognitive science. Overall, the subject of Explainable AI is still developing quickly due to the urgent demand for accountability and transparency in AI systems across numerous domains and applications. To increase the trust and usability of AI technology, researchers are always attempting to build more efficient and comprehensive XAI methodologies.


1.5	Approach

The data for both the use cases are sourced from different online sources. The deepfake images along with real images are obtained for the CNN model and the tweets from twitter are sourced along with its respective classifications. The images are preprocessed into the correct size and, the tweets are all cleaned to remove unwanted characters and then tokenized. Different architectures are built and fitted to the data and then tested to find the best accuracy. Many preexisting architectures are also fitted to see if it performs better than the custom-built models. The best models are selected for each use case and then that model is integrated into the application back-end. The web based front-end of the application is designed to obtain the input from the user and to pass on the details to the server where the corresponding model is picked and run. Then the output along with its explanations are passed back to the web client to display on the screen. The explanations will give a detailed visualization of the classification logic, which will help understand why the model came to a specific decision of classifying the image as fake or real for one use case and the classification of tweets as positive, negative or neutral in other use case.


1.6	Thesis Structure
The structure of the thesis is arranged in the following order:

•	Chapter 2 comprises of the important terminologies required to understand the technical concepts. It also contains the literature review which depicts the projects research carried out specifically in the field of deep learning classification and explainable AI.
•	Chapter 3 comprises of the workflow of the project. It gives information on the software, libraries and tools used along with the methodology that was followed for this thesis.
•	Chapter 4 comprises of the implementation of the project. This includes the code snippets and the reports.
•	Chapter 5 comprises of the results obtained from the implementation and the screenshots of the results
•	Chapter 6 consists of the conclusion for the project and the future work that can be incorporated in this field.
















2. Literature Review


2.1	Background

Artificial intelligence (AI) [11] has been a subject of extensive research and development, aiming to replicate human-like cognitive abilities in machines. [Nilsson, 1998] emphasizes the importance of symbolic AI, a paradigm where knowledge is encoded via symbols and rules, at an early stage in AI development. The essential method of describing expertise through symbolic representations is clarified in this landmark work [14], establishing the foundation for the development of expert systems [Shortliffe, 2014]. Despite its potential, this strategy struggled to address the complicated, convoluted nature of real-world issues. The exploration of Nilsson, which emphasizes the fundamental value of symbolic representation while admitting the practical challenges that sparked the development of AI approaches, is a turning moment in the history of AI. However, challenges arose due to the complex nature of real-world problems.

The expansive evolution of Artificial Intelligence (AI) spurred the ascendancy of Machine Learning (ML) [10], ushering in an era where algorithms attain knowledge from voluminous data streams. Classical ML algorithms, such as decision trees [2] and support vector machines [17] (Breiman et al., 1984; Vapnik, 1995), are at the vanguard of this shift. These algorithms aimed to reveal complex patterns within datasets to provide predictive and analytical capabilities. Traditional neural networks, on the other hand, saw a decline in popularity as the AI landscape changed because of scaling and training complexity issues. 

In the midst of this paradigm shift, [LeCun et al., 1998] brought forth Convolutional Neural Networks (ConvNets) [8], infusing a revitalized vigor into the realm of neural networks. The scientific community was reenergized by this seminal introduction, which led to a reconsideration of the potential of neural networks. ConvNets excelled in identifying patterns and features within complicated datasets, particularly ones encompassing visual information, and they were crucial in overcoming obstacles that had previously stymied research. This upsurge signaled a turning point that ultimately sped up the development of the revolutionary idea of deep learning. ConvNets and the ultimate creation of deep learning were made possible by the transition from traditional AI methods to the advent of ML, which was aided by decision trees and support vector machines. The trip is evidence of AI's dynamic nature because new approaches build on the foundations of their forerunners, each change signifying a step toward more powerful and effective AI paradigms.

Deep Learning [5], a revolutionary subset of Machine Learning (ML), developed as a vibrant area propelled by deep neural network technology. The idea of deep belief networks was pioneered by this evolution and a new era of unsupervised learning by [Hinton et al., 2006]. This addition is significant because it can identify complicated patterns in data without explicitly labeling them; this skill holds considerable potential for automating feature extraction and improving comprehension of difficult data structures. However, the groundbreaking work of [Krizhevsky et al., 2012] marked the real turning point in deep learning. Their ground-breaking Convolutional Neural Network (ConvNet), dubbed AlexNet [6], confounded expectations by excelling in image classification tests like never before. Prior benchmarks were shattered by AlexNet's impressive performance in the ImageNet Large Scale Visual Recognition Challenge, which also paved the way for the widespread use of deep neural networks in visual recognition applications. This significant accomplishment sparked renewed interest in neural networks and served as the tipping point that propelled Deep Learning's rapid development.

The Long Short-Term Memory (LSTM) [15] networks, which were developed by [Sutskever et al., 2014] in combination with improvements in image categorization, are a crucial achievement in the Deep Learning field. These specialized recurrent neural networks are particularly well-suited for tasks involving time series data, language modeling, and more since they are excellent at capturing sequential dependencies. Traditional recurrent neural networks had a vanishing gradient problem that LSTMs were able to solve, and their success in sequence modeling paved the way for innovations in speech recognition, machine translation, and sentiment analysis.

Additionally, the development of deep learning went beyond pictures and sequences. The Transformer architecture, developed by [Vaswani et al., 2017], revolutionized Natural Language Processing (NLP) [18] by introducing attention methods. The Transformer's self-attention mechanism allowed the model to recognize global relationships inside sequences, making it very useful for language production and machine translation tasks. Recurrent architectures were replaced with attention-based architectures, which enabled models to interpret longer sequences more effectively and precisely.

The inventiveness of researchers who realized the potential of deep neural networks was what fueled Deep Learning's rise within the ML environment. Deep belief networks, ConvNets, LSTMs, and attention mechanisms are examples of how each milestone improved the capabilities of the discipline. Deep Learning's dynamic path highlighted the transformative impact of utilizing multi-layered neural networks to release previously unattainable insights and capabilities within complicated data domains, from image classification to sequence modeling and NLP.

Explainable Artificial Intelligence (XAI) [13], a crucial approach aiming at improving model transparency and interpretability, has emerged as a result of the explosion in complicated machine learning models and their inherent "black-box" character. Researchers have painstakingly attempted to shed light on the decision-making processes that transpire within these models' various layers in response to the challenges they pose. [Ribeiro et al., 2016] made a significant addition to the field of XAI by developing the Local Interpretable Model-agnostic Explanations (LIME) framework. This approach cleverly used the perturbation principle to solve the problem of explaining black-box models. LIME provided a revolutionary method that involved perturbing input instances and carefully monitoring the changes in output that resulted. This approach made it possible to create locally accurate surrogate models that provided clear insights into the correlation between inputs and outputs. LIME gave practitioners the capacity to understand the driving forces behind model outputs and decipher forecasts by precisely approximating the decision bounds of complex models.

Continuing along this path, [Lundberg & Lee, 2017] proposed the Shapley Additive Explanations (SHAP) [9] values, a paradigm shift that created a consistent framework for determining the relevance of features. This approach was based on the principles of cooperative game theory and was motivated by the idea of an equitable distribution of rewards among players. A mathematical foundation for attributing the contributions of specific attributes to model predictions was introduced by SHAP values. In addition to offering a more comprehensive knowledge of feature impact, this principled approach also cleared the way for cutting-edge visualization approaches that shed light on the interactions between inputs and model outputs.

[Guidotti et al., 2019] undertook a thorough survey that crossed the spectrum of methodologies in their quest to fully grasp the landscape of XAI methods. The results showed a rich tapestry of strengths and limitations. This groundbreaking work [4] highlighted the complexity of XAI by incorporating a wide range of methodologies, from perturbation-based methods to rule-based justifications. By analyzing these methods' effectiveness in a variety of circumstances, the authors gave practitioners a nuanced perspective that helped them choose the best methodology for their particular objectives.

Additionally, [Dhurandhar et al., 2018] started a novel investigation by concentrating on contrastive explanations, a trailblazing direction that strengthened the toolbox of interpretability tools. Through highlighting perturbations that resulted in diverse expectations, their study [3] added a transformative dimension. The researchers demonstrated perturbations that greatly altered the model's decision-making process by contrasting cases that produced differing model outputs. This fresh perspective not only broadened the range of XAI approaches but also captured the complex dynamics of predictions, providing previously hidden insights.

These trailblazing efforts have built a solid basis for the XAI environment in the ever-growing search for transparency and accountability in AI. The development of LIME, SHAP values, thorough surveys, and the investigation of opposing theories all collectively shed light on the approach to solving the mystery of "black-box" models. These approaches combine to provide a link between sophisticated algorithms and human understanding, making it easier to share the knowledge and understanding that are essential for promoting confidence and acceptance in the field of AI-driven decision-making.


2.2	Lime Method

The development of Local Interpretable Model-agnostic Explanations (LIME) [13], which bridges the gap between the interpretability requirement of humans and the opaqueness of complicated machine learning models, is crucial to the study of Explainable Artificial Intelligence (XAI). As a result of the advanced models' intrinsic "black-box" nature, LIME emerges as a methodological innovation that clarifies the decision-making procedures buried in these complex architectures.

LIME offers a fresh method for understanding the mysterious behavior of black-box models because it is based on the perturbation principle at its core. The key to LIME is its capacity to produce locally accurate surrogate models, which reduce complex predictions into understandable explanations. This is achieved by deftly manipulating the input instances: LIME modifies certain data features or properties while rigorously tracking how the model outputs change as a result. The basic decision boundaries of the original model are effectively approximated by LIME by creating a connection between perturbed inputs and output changes.
 
Figure 2.1: Lime Workflow

The localization of the interpretability attained with LIME is what makes it distinctive. LIME concentrates on defining how predictions develop within a particular, limited region of the input space rather than attempting to understand the mechanics of the model as a whole. This local scope is especially important in situations where models behave differently depending on where in the input space they are located.

The selection of an instance to be explained is the first step in the LIME explanation generation process. A distribution centered on the original instance is then used to choose perturbed instances. The prediction of the model is observed for each disrupted instance, enabling the creation of a substitute interpretable model that behaves in a manner similar to the original model. This substitute model, which is frequently more straightforward and straightforward than the original, represents the underlying decision-making process near the chosen case.

When it comes to machine learning models, LIME provides a flexible toolkit that can accommodate everything from sophisticated deep learning architectures to conventional classifiers. As it can be applied to any black-box model without the requirement for model-specific insights, it is consistent with XAI's model-agnostic nature.

LIME also improves the fairness and robustness of the model. It facilitates the identification of biases and probable sources of model mistakes by helping practitioners to pinpoint the features that have the biggest impact on predictions. It turns into a crucial tool for spotting and fixing bad behavior in AI systems.

LIME essentially acts as a crucial intersection where the world of opaque models meets the need for transparency and accountability in society. By using perturbation to produce interpretable surrogate models, it elegantly bridges the conceptual gap between complex machine learning algorithms and human comprehension. LIME converts complex model outputs into understandable insights through its locally true explanations, encouraging trust, responsibility, and comprehension in the field of AI decision-making.



3. Design, Data and Methodology

In this chapter the technical methods used are briefed so that the readers can easily understand the terminologies and the functionalities used. The workflow along with the design details used in the project are also mentioned for better understanding.


3.1	Tools and Techniques

Visual Studio Code:
Visual Studio Code (VS Code) is a free and open-source code editor developed by Microsoft. It has gained widespread popularity among developers due to its lightweight, extensible, and highly customizable nature. VS Code supports a wide range of programming languages and offers numerous features that make it an excellent choice for Python development.
Python is the programming language used extensively in this project. It is a relatively easy language to learn thanks to its clear and understandable grammar. The task is finished with less lines of code. Because of its great scalability, versatility, large number of packages and modules, support for graphics and visualization, fantastic community, and abundance of easily available resources, Python is widely used in data analysis. 
In this research the main libraries used for application building are listed below:
•	Flask: Flask is a lightweight micro web framework for Python, allowing developers to build web applications and APIs quickly. It offers routing, HTTP method support, templates, and request/response handling. Its simplicity and flexibility make it popular for small to medium-sized projects. Flask can be extended with various community-built extensions.
•	Pandas: Python’s Pandas library is used to modify data sets. It provides resources for exploring, organizing, analyzing, and manipulating data. We can investigate enormous data sets with Pan- das and make inferences based on statistical concepts. Data sets that are disorganized can be readably and practically usefully organized by pandas.
•	NumPy: Python’s NumPy library offers a wide range of capabilities, including extensive mathematical functions, random number generators, linear algebra functions, Fourier transforms, and more. The de facto standards for array computation at the moment are the NumPy vectorization, indexing, and broadcasting concepts since they are rapid and adaptable. NumPy is usable and effective by programmers of many backgrounds and degrees of experience because of its high-level syntax.
•	Matplotlib: Python’s Matplotlib toolkit is a comprehensive tool for creating static, animated, and interactive. Matplotlib makes both basic and complicated tasks possible. It can create interactive charts that can zoom, pan, and update. And even the layout and visual style can also be modified.
•	Seaborn: Seaborn is matplotlib-based Python data visualization library is. It offers a sophisticated drawing tool for creating eye-catching and informative statistical visuals.
•	Scikit-learn: Sklearn is a popular machine learning library for Python. It provides a wide range of tools and algorithms for data preprocessing, classification, regression, clustering, and much more. With an easy-to-use interface, sklearn is the perfect choice for both beginners and experienced data scientists.
•	TensorFlow: TensorFlow is an open-source library developed by Google especially for deep learning applications. TensorFlow was first developed to perform massive numerical computations without having deep learning in mind.  Google made it open source because it turned out to be quite beneficial for the advancement of deep learning as well. TensorFlow only accepts tensors, which are multi-dimensional arrays with additional dimensions. Multi-dimensional arrays are very useful when working with large amounts of data. The basis for TensorFlow's operation is node-and-edge data flow graphs. Because the TensorFlow execution mechanism takes the form of graphs, it is easier to distribute the execution of TensorFlow code utilizing GPUs across a cluster of computers.
•	Keras: To implement neural networks, Google created the high-level Keras deep learning API. It is created in Python and is designed to simplify the implementation of neural networks. Numerous backend neural network computations are provided as well. Keras is fairly easy to learn and use since it provides a high level of abstraction Python frontend and the option of multiple back-ends for computation. As a result, Keras is slower than other deep learning frameworks but far more user-friendly for beginners.

The application is developed as a web-based client server model. The server-side code is entirely created with the help of the above mention python and its libraries. The client-side code is built using the below.

•	JavaScript: JavaScript is a dynamic scripting language used for web development. It enables interactive web elements, handling events, and modifying content on-the-fly, making websites more engaging and responsive.
•	CSS (Cascading Style Sheets): CSS is a style sheet language used to control the layout and presentation of HTML documents. It defines how HTML elements should be displayed, making it easy to change fonts, colors, spacing, and other design aspects.
•	HTML (Hypertext Markup Language): HTML is the standard markup language for creating web pages. It provides the structure and content of a webpage by defining elements such as headings, paragraphs, links, images, and more.


3.2	Workflow and Design

The figure 3.1 describes the design and research workflow. The project is broken down into seven parts to provide structure and manageability. Gathering information from online sources is the first step. The second stage is on examining the distribution of various classes in the dataset. Pre-processing the data to make it appropriate for each of the models is the third stage. The individual training and evaluation of each model is the emphasis of the fourth step. The fifth step is choosing the best performing model and coding the explanation for each use case and plugging it to the application server code. The sixth step is to build a graphical user interface (web page) which acts as the client side, where the user can interact with the application and view the inputs and outputs. The final step is concerned with integrating the client-side and server-side to seamlessly exchange the user input from client to server and get the results for the respective model run on the server side back to the client side for the user to view.

	 

Figure 3.1: Workflow






3.3	Data Collection and Description

The fake and real images data used for this CNN model is created by [Bird, J.J. et al. 2023]. CIFAKE:AI-Generated Synthetic Images [1]. CIFAKE is a dataset that contains 60,000 synthetically-generated images and 60,000 real image.
The dataset contains two classes - REAL and FAKE. For REAL, the images were collected from Krizhevsky & Hinton's CIFAR-10 dataset. For the FAKE images, the equivalent of CIFAR-10 with Stable Diffusion version 1.4 was used to generate it. There are 100,000 images for training (50k per class) and 20,000 for testing (10k per class)

 
Figure 3.2: CIFAKE Images

The text classification model will use the Bitcointweets dataset [16] created by [TEESOONG. 2020]. The dataset is based on the tweets generated related to bitcoins in twitter. The manual classification of the tweet classes has been made by the author and provide the data in a csv format. The data contains many special characters and tags which needs to be cleaned as a preprocessing step before the text content can be used for classification. The wordcloud of the most frequent words in the entire dataset has been depicted below.

 
	Figure 3.3: Tweet text wordcloud


3.4	Methodology

The aim of this project is to create an application which can take in user input as images or text and then do the classification of images to fake or real and the classification of text to positive, negative or neutral. In the case of images, the goal is to find the best architecture for the CNN model which can classify the images with best possible accuracy. Different combinations of layers and various hyperparameters will be considered for building the best possible CNN model. Different combinations will be tried out to understand the performance of the model for the same images on different proven architectures [6] like VGG16, MobileNet, RestNet50 and DenseNet to see if these can be plugged-in using transfer learning into the base layer of the CNN model. The best accuracy can be achieved for any of the above architectures when chosen as the base layer along with carefully selected hyperparameters for the rest of the layers in the custom-made CNN [7]. This architecture will be used for the image classification. In order to achieve the best results for the text classification of tweets using natural language processing, the LSTM (Long Short-Term Memory) model [12] will be custom-built using the best possible hyperparameters after rigorous tuning. 
After the models are built, the explanations for these models are generated with the help of Lime package. LIME (Local Interpretable Model-agnostic Explanations) is an explainable AI technique designed to provide insights into the predictions made by complex machine learning models. It works by approximating the behavior of a black-box model in a local region around a specific instance to make its predictions more interpretable. LIME creates a straightforward storyteller that learns from significantly altered queries and describes the patterns it discovered in order to help you understand the "why" behind a black-box response. Remember that LIME doesn't provide the exact response that the black-box would provide; rather, it aids in your understanding of the broad reasoning behind the black-box's approach to a given question.

































			      Figure 3.4: Lime Algorithm 



Here's a step-by-step algorithmic explanation of how LIME [13] works:
1.	Select a question (input) for which you wish to understand the reasoning behind the black box's response.
2.	Generate a random perturbation of the instance. You produce slightly varied iterations of your input, or query. Make sure that it has features that are similar to (for example, by changing only a few features while maintaining others). These variations are somewhat modified versions of your original query, like "what if" possibilities. Use the black-box model to record the perturbed sample and its accompanying label.
3.	Create perturbed samples from the instance, and then encode them into a format that interpretable models can use. This might involve converting categorical variables, text, or images into numerical feature representations.
4.	Select a straightforward, comprehensible model (such as a decision tree or linear regression) that can simulate the behavior of the black-box model in the vicinity of the instance. To train the interpretable model, use the encoded instances and their related labels.
5.	Based on their resemblance to the instance and/or their proximity in the feature space, give the perturbed samples weights. By doing this, it is made sure that the model concentrates more on samples that are more like the instance.
6.	The black-box model's decision-making for the instance can now be understood through analysis of the trained interpretable model. The interpretable model's feature weights, coefficients, or learnt rules can be utilized to justify the black-box model's conclusion in this case. The interpretable model's coefficients, also known as feature importance values, show how much each feature contributed to the instance's prediction. Values that are positive or negative indicate the influence's direction.


















4. Implementation


4.1.	Creating the CNN Model for Image Classification


The images data has been gathered from CIFake. Next the CNN model needs to be created for classification of fake images. Here is a brief overview of the CNN.


Figure 4.1: CNN Architecture [19]



Convolution Neural Networks

Convolutional neural network (CNNs) [19] is one of the main categories of neural networks to perform the tasks of image recognition and classification. Object detection, facial recognition, and other areas are among the ones where CNNs are frequently applied. An image is processed using CNN classifiers after being supplied. Depending on the image's resolution, computers interpret input images as pixel arrays. H x W x D (h = height, w = width, and d = dimension) will appear depending on the image's resolution. For instance, consider a 128 × 128 x 3 RGB matrix image, where 3 denotes RGB values. Each input image must go through a sequence of filter (kernels), convolution layers, pooling, fully connected layers (FCs), and apply SoftMax function to identify an object with probabilistic values between 0 and 1 in order to train and test the CNN models of deep learning.

Convolution Layer

The first layer where features from an input image are extracted is convolution. Convolution preserves the relationship between pixels by learning visual attributes from small input data squares. This mathematical operation takes two inputs: an image matrix and a kernel, or filter. Convolution of an image with different filters by adding filters will be used to carry out edge detection, blur, and sharpening techniques.

Padding

Usually, the filter does not precisely match the input image. Our options are two:

•	To make the image match, zero-padding the zeros.

•	Remove the portion of the photo where the filter does not match. True padding is what is referred to as partial preservation of the image.


ReLU Activation Function

ReLU stands for Rectified Linear Unit for a non-linear operation {f(x) = max (0, x)}.
ReLU is designed to add nonlinearity to our model. Since real-world data would prefer that our model learn, non-negative linear values would be appropriate. In addition to using tanh or sigmoid, ReLU can also be substituted with other nonlinear functions. ReLU is used by many data scientists since it outperforms the other two in terms of performance.


Pooling Layer

When the number of image pixels are too high, the section of pooling layers will reduce the number of parameters. Spatial pooling, also known as subsampling or down sampling, reduces the dimensionality of each map while maintaining crucial features. Spatial pooling comes in a variety of forms:
•	Max Pooling

•	Sum Pooling

•	Average Pooling 


Dropout

Although dropout is frequently used to regularize deep neural networks, the ways in which it is implemented on fully connected layers and convolution layers are fundamentally different.

Fully Connected Layer

In this layer, we convert our matrix into a vector and input it into a fully connected layer to simulate a neural network.



The CNN architecture will be built using the above components. But to improve the model, a technique called transfer learning can be used. Transfer learning is using knowledge from one task to improve a related task. Initial learning occurs on a source task, then the model adapts to a target task with fine-tuning. This boosts performance, especially with limited target data.
Different combinations were tried out to understand the performance of the model for the same images on different proven architectures like VGG16, MobileNet, RestNet50 and DenseNet to see if these can be plugged-in using transfer learning into the base layer of the CNN model.



Models	Accuracy
VGG16	87%
MobileNet	79%
DenseNet	77%
ResNet	75%
Custom made CNN	65%

Table 4.1: Model Selection


Based on the different combinations tried, VGG16 gave the best results when tried to classify fake images of size 128 * 128.  So, the base layer of the CNN model was fit with a pre-built VGG16 architecture and the weights used were those of ‘ImageNet’. Now the remaining layers of the model were custom built.  

Here is the model architecture.

 

Figure 4.2: CNN Model Summary
Different combinations of hyperparameters were used to carefully select the best parameter values for the model. The code snippet used to build the model is shown below:


 

Figure 4.3: CNN Model Building
 
4.2.	Creating the LSTM Model for Text Classification

The text classification model was built using a LSTM architecture which is a form of recurrent neural network. 


 

   Figure 4.4: LSTM Architecture

Long Short-Term Memory (LSTM) [12] is a recurrent neural network architecture designed by Sepp Hochreiter and Jürgen Schmidhuber in 1997.
The memory unit, also known as the LSTM unit, is the only component of the LSTM architecture. There are four feedforward neural networks in the LSTM unit. There are two layers in each of these neural networks: the input layer and the output layer. Input neurons are linked to all output neurons in each of these neural networks. The LSTM unit thus comprises four completely linked layers.
The selection of information is carried out by three of the four feedforward neural networks. The forget gate, the input gate, and the output gate are what they are. These three gates are used to carry out the three common memory management operations: erasing data from memory (the forget gate), adding new data to memory (the input gate), and using data already in memory (the output gate).
The fourth neural network, the candidate memory, is used to create new candidate information to be inserted into the memory.

Input and Output

Three vectors (three lists of numbers) are input into an LSTM unit. Two vectors were produced by the LSTM at the previous instant (instant t-1) and came directly from the LSTM itself. The hidden state (H) and the cell state (C) are these. The third vector is external. This is the vector X that was sent to the LSTM at instant t, also known as the input vector.

The LSTM governs the internal flow of information through the gates and alters the values of the cell state and hidden state vectors given the three input vectors (C, H, X). vectors that will be included in the LSTM input set at moment t+1. The concealed state serves as a short-term memory, while information flow control is used to make the cell state serve as a long-term memory.

In actuality, the LSTM unit updates the long-term memory (cell state) using both fresh outside information (the input vector, X) and recent past information (the short-term memory, H). In order to update the short-term memory (the hidden state, H), it lastly uses the long-term memory (the cell state, C). The output of the LSTM unit in instant t is also the hidden state identified in that instant. It is what the LSTM offers to the outside world in order to carry out a particular mission. In other words, it is the behavior that determines how well the LSTM performs.

Gates

Information selectors are the three gates (forget gate, input gate, and output gate). They are tasked with making selector vectors. A vector containing values between zero and one and close to these two extremes is referred to as a selector vector.

Input Gate: Decides what new information to add to the cell state. It considers the current input and the previous hidden state.

Forget Gate: Determines what information to remove or forget from the cell state. It considers the current input and the previous hidden state.

Output Gate: Controls what information from the cell state should be used to compute the output or prediction for the current time step.

Candidate Memory: A candidate vector, or a vector containing information that could be added to the cell state, is created by the candidate memory. The hyperbolic tangent function is used by potential memory output neurons. The characteristics of this function guarantee that the candidate vector's values are all between -1 and 1. The information that will be added to the cell state is normalized using this method.

Backpropagation Through Time

A neural network's output Y is dependent on an information flow that travels through numerous elements arranged in a chain. Each of these components is constructed in such a way that even a minor increase in its output value will have an impact on succeeding components up to the output of the network (Y). By calculating the relationship between the rise in an element's output value and the rise in network error, the mistake is minimized. This process is referred to as backpropagation.
The error gradient can be calculated using backpropagation. The concept of derivative is generalized by the gradient for functions with various inputs. The idea of a ratio between (instantaneous and infinitely small) increments is formalized by the concept of a derivative. The chain rule is a mathematical approach that is used in backpropagation. The chain rule is based on the idea that if an error grows twice as quickly as an increase in Y and once as quickly as an increase in D, then the mistake E must grow four times as quickly as an increase in D. Two ratios (two derivatives) are multiplied to arrive at this result.
In RRNs, information does not always travel via neural network components. It also develops gradually. The error made by the network at time t is also influenced by the data gathered from earlier periods of time and processed at these points in time. Backpropagation consequently takes into account the network of dependencies between time instants in an RRN. It is known as Backpropagation Through Time (BPTT) for this reason.

Here is the model architecture.


 

  Figure 4.5: LSTM Model Summary



Different combinations of hyperparameters were used to carefully select the best parameter values for the model. The code snippet used to build the model is shown below. The first image is the model definition and the required packages being imported. Then the data is read and cleaned.
The next image is the code where the text data is tokenized using a sequencer and padded to be of the same length. This is defined as separate classes and is required for the LSTM to classify the sentence correctly. So here the lstm model, the padder class and the sequencer classes are all integrated together to work in pipeline.
 

 Figure 4.6: LSTM Model Creation

 

Figure 4.7: LSTM Model Pipeline



4.3.	Creating the Flask Server side Application (for Integration of the Models)

The models created were then integrated seamlessly with the help of Flask. Flask is a lightweight Python web framework that aids in swiftly building web apps and APIs. Its simplicity and customization capabilities make it advantageous for deploying machine learning models. Flask facilitates model deployment as web services, supports API creation for seamless communication, and enables interactive web interfaces to showcase predictions. Its integration with other libraries enhances functionality, while its scalability suits a range of applications. With an active community and user-friendly nature, Flask is ideal for transitioning ML models into accessible web-based tools, enhancing user engagement and interaction without extensive coding knowledge.

Routes: These define the URLs and corresponding functions that handle requests. Using decorators, you associate functions with specific URL paths.

Views: Views are the functions that handle the requests and return responses. They generate content that the user sees in the browser.

Templates: Flask supports the Jinja2 templating engine, allowing you to create dynamic HTML templates that can be rendered with data.

Static Files: These include CSS, JavaScript, and image files that are served directly to users' browsers for styling and interactivity.

Request and Response Objects: These objects provide access to data from incoming requests and allow you to construct responses.

The CNN Model and the LSTM Model are saved as executables and then loaded in the flask application code. See below the required libraries imported and the models being loaded.

 

Figure 4.8: Flask Model Loading

There are 3 routes defined in the code which dictate the flow of execution of the application. Routes in Flask are crucial because they link particular URLs to functions, allowing web applications to respond appropriately to various requests. They offer an organized mechanism to specify which actions or content go with different URLs, making the application navigable and useful. Developers may structure code, divide concerns, and construct dynamic, interactive web pages thanks to this abstraction. Flask improves user experience by streamlining the process of responding to user inputs and making it simple to build feature-rich web apps by connecting routes and functions.

•	/ or /Home route

In the home route the client-side web page was rendered using the respective html code. This will enable the front-end users to view the web page and connect the backend server code with the front-end client side.

 

Figure 4.9: Home route



•	/Predict route

This route handles the prediction of the uploaded image or tweet. Upon receiving an image, the function analyzes it, determines the image's authenticity using a CNN model (cnn_model), and then returns a JSON message stating whether the image is real or fake. If a tweet is given, the function predicts the sentiment of the tweet using an NLP pipeline (nlp_pipeline) and returns a JSON message with the sentiment forecast.
 

Figure 4.10: Predict route


•	/Explain route

The XAI part of the application is handled in this route. It generates explanations using LIME (Local Interpretable Model-Agnostic Explanations). If an image is provided, the function processes it, generates an explanation for the CNN model classification, visualizes the explanation, and returns the explanation image as a response. If a tweet is provided, the function uses LIME to explain the sentiment classification and returns the explanation image.



 


Figure 4.11: Explain route



4.4.	Creating the Flask Client side Web User Interface 


The client-side of the flask application is a web user interface where the user can choose the model, provide inputs and the application will run and provide the respective classification results along with the detailed explanation of why the model gave a particular result or the models reasoning behind its conclusions. 
The client-side code consists of 3 parts:

•	HTML

Since HTML is the industry-standard markup language for creating and organizing web content, Flask uses it to create webpages. The appearance and user interface are handled by HTML, while the data and logic are handled by the backend framework Flask. HTML makes it simple to render dynamic material and interact with visitors by allowing developers to specify the structure, layout, and elements of web pages. Developers may construct cohesive and interactive web apps that offer a user-friendly experience, separate content from design, and assure compatibility across many devices and browsers by combining the backend capabilities of Flask with the frontend capabilities of HTML.

These capabilities are used to create the HTML code of the webpage for the application. The page includes a header with the application title and a dropdown menu (<select>) that allows users to choose between two machine learning models. Each model option contains a description of the model's functionality and contribution.

The main content area is divided into sections. The input section contains a file input element (<input type="file">) for uploading an image, and an image container to display the selected image. There's also a hidden text area element for entering a tweet.

The output section displays the results of the selected model along with explanations. It includes a loading message while the explanations are being generated. An image element (<img>) shows visualized explanations, and a <div> element displays the output message.

At the bottom, there's a "Run the Model" button that triggers a JavaScript function (runModel()) when clicked.

External CSS and JavaScript files (styles.css and script.js) are linked using Flask's url_for function to provide styling and interactive behavior to the webpage. The JavaScript script handles dynamic interactions, such as model selection, image display, and running the model.

In summary, this HTML code creates a user interface for selecting machine learning models, uploading images or entering text, and viewing model results and explanations. JavaScript is used to enhance interactivity and dynamic behavior on the webpage.

 

Figure 4.12: HTML Code




•	CSS

Since CSS (Cascading Style Sheets) separates the appearance and layout from the content defined in HTML, Flask uses it for website design. While CSS improves the aesthetic appeal, formatting, and styling of web pages, Flask manages backend functions. Developers may adjust colors, fonts, spacing, and responsiveness across devices while producing aesthetically pleasing designs that are consistent and readable. This division of responsibilities speeds up development and enables productive cooperation between designers and developers. Because of Flask's interaction with CSS, frontend functionality complements backend functionality, resulting in well-structured and aesthetically pleasing online applications.

 

Figure 4.13: Snippets of the CSS Code

The webpage for the application is styled using the specified CSS code. It controls the typeface, background color, alignment, and spacing of different elements. Input components like file upload and text area are stylized for appearance and interaction, while headers and titles are prepared. Flexbox and grid technology are used in the layout, which has sections for input and output content. Buttons feature hover effects, and loading messages are configured. Output content is designed for a unified display, including the description, image, and message. Overall, this CSS improves the website's readability, interactivity, and aesthetic coherence for a user-friendly experience.

•	JavaScript

JavaScript is utilized in Flask for webpages to introduce dynamic and interactive features to the frontend. JavaScript gives web applications real-time answers, dynamic content updates, and improved user interaction while Flask handles the backend logic. In addition to enabling asynchronous connection with the server for quicker data retrieval without reloading the entire page, it enables developers to build animations, validate forms in real-time, and retrieve data more quickly. React or Vue.js are JavaScript frameworks that can be used to further improve user interfaces and enable component-based modular design. Developers may create dynamic, responsive, and user-friendly web apps that go beyond static content presentation thanks to the integration of Flask's backend functionality with JavaScript's frontend interactivity.

The JavaScript code is instrumental in creating the functionality of the application webpage. It orchestrates various interactive elements and processes to offer users an engaging experience:

 

Figure 4.14: Interactions Code

Image Upload Interaction: The code establishes an event listener for the image input element (imageInput). When a user selects an image, the FileReader is employed to read and convert the image data into a URL. This URL is then assigned to the src attribute of the selectedImage element, displaying the chosen image on the webpage.

Model Selection Interaction: Another event listener is attached to the model selection dropdown (modelSelect). When a user chooses a model, the event listener is triggered. Based on the model selection, the code determines whether to display the image upload container (imageContainer) or the tweet input container (tweetContainer). This responsive behavior enhances user interaction by showing relevant input options for the selected model.

Run Model Function: The runModel function is at the core of the model prediction and explanation process. When the "Run the Model" button is clicked, this function is invoked. It retrieves the selected model and its associated ID from the dropdown. If the chosen model is image-based, it sends the selected image to the server for prediction. If an NLP-based model is selected, the function sends the entered tweet to the server. The responses, which include predictions and explanations, are then displayed in the output section.

 

Figure 4.15: Run Model Function
Loading Messages: The code implements loading messages while waiting for the server's response. This keeps users informed about ongoing processes and ensures a smooth and informed user experience.

So, this JavaScript code significantly contributes to the functionality and interactivity of the application webpage. It enables users to upload images, select models, and view model predictions and explanations in real-time. The code's modular structure and event-driven approach create a seamless and user-friendly environment, enhancing engagement and facilitating the interaction between users and machine learning models.

In summary, when a user uses the website, they can select a machine learning model, add text or photos, and then click the "Run the Model" button. The Flask application interacts with the proper machine learning model, processes the input, and offers predictions and explanations based on the model that was chosen. Users can view the results and learn more about the model predictions in the output area of the website.

In essence, this integrated system gives users a user-friendly, responsive interface via which they may interactively explore predictions and explanations for image and text inputs from machine learning models.






















5. Results

The application gets hosted on the web where the user can interact with the application to pick the model for which the user wishes to do the classification and get the results along with the explanations. First the user has to choose the model to run. Here the two models are given in a drop down and the user can choose one.

 

Figure 5.1: Model Selection

Once each model is selected the respective screen for entering the required user inputs are shown. 


Below few of the use cases have been showcased.

•	CNN Model for fake image classification

When the CNN Model is selected the below screen appears and the user can upload an image which needs to be checked for Fake or Real image classification. 
  

Figure 5.2: CNN Model Window


	Scenario 1: Fake Image:

The fake image which was created using deepfake was chosen and uploaded. Then the run model was clicked which transferred the image to the server-side model code for CNN. The image was run in the model and it classified it as Fake. Then the explanations for the classification was done by the LIME (Local Interpretable Model-agnostic Explanations) package. 

In order to classify fake images, LIME generated local explanations for each prediction using the CNN model. It used a method of explanation that was image-specific. Masking regions of interest in the input image caused LIME to tamper with it, and it then measured the effect on the model's output. Red and green highlighting, respectively, were used to indicate the positive and negative contributions of various image sections. These highlighted areas reflected characteristics that affected the model's prediction of whether an image was real or false. Users could understand why the model labeled an image as phony because to LIME's interpretable insights into CNN's decision-making process, which also assisted in identifying potentially altered areas.
	


 

Figure 5.3: Fake Image Classification 
and Explanation

Scenario 2: Real Image:

The real image was chosen and uploaded. Then the run model was clicked which transferred the image to the server-side model code for CNN. The image was run in the model and it classified it as Fake. Then the explanations for the classification was done by the LIME (Local Interpretable Model-agnostic Explanations) package. 

Using its explanation methodology, LIME identified real images for the CNN model. LIME used the input image as a starting point, perturbed it, and then examined the influence on the model's predictions. It then used red to emphasize the regions that made good contributions. These areas indicated characteristics that helped the program choose whether to categorize an image as real. LIME offered interpretable insights into why the CNN saw an image as authentic by pointing out instances where the model's response moved in favor of the real class. This strategy allowed people to comprehend the CNN's thought process and decision-making process when classifying images as real.
	


 

Figure 5.4: Real Image Classification 
and Explanation


•	LSTM Model for text classification

When the LSTM Model is selected the below screen appears and the user can enter text or tweet which needs to be checked for Positive, Negative or Neutral classification in the text box which appears. 
 

 

Figure 5.5: LSTM Model Window


	Scenario 1: Positive Text:

The positive tweet on cryptocurrency from twitter was chosen and entered. Then the run model was clicked which transferred the text to the server-side model code for LSTM. The image was run in the model and it classified it as Fake. Then the explanations for the classification was done by the LIME (Local Interpretable Model-agnostic Explanations) package. 

Using its explanation methodology, LIME develops a more straightforward model that roughly mimics the LSTM's decision-making process in order to explain positive text classification in LSTM tweet categorization. LIME accomplishes this by creating different perturbed copies of the input tweet and observing how these modifications affect the LSTM's predictions. LIME captures the links between words and categories by training a surrogate model on the perturbed data. LIME finds the words and phrases that were most responsible for the positive classification of a text. An explanation that grants relevance scores highlights these powerful words. High scores for adjectives like "happy," "exciting," or comparable keywords suggest that they have a favorable affect. LIME's explanation provides a clear view of how the LSTM's intricate analysis of language elements leads to the conclusion that the text is expressing a positive sentiment.
	

 

Figure 5.6: Positive Text Classification 
and Explanation


Scenario 2: Neutral Text:

The neutral tweet on cryptocurrency from twitter was chosen and entered. Then the run model was clicked which transferred the text to the server-side model code for LSTM. The image was run in the model and it classified it as Fake. Then the explanations for the classification was done by the LIME (Local Interpretable Model-agnostic Explanations) package. 

Using its explanation methodology, LIME clarifies the neutral text classification in LSTM tweet classification. LIME accomplishes this by generating altered versions of the input tweet and examining how these affect the predictions made by the LSTM. These altered instances are then used to construct a surrogate model that attempts to simulate the behavior of the LSTM. LIME determines the crucial words and phrases that are most responsible for the neutral classification of a text. These key terms are highlighted along with significance scores in the explanation. If terms like "average," "standard," or similar expressions score higher, it suggests that they had a big part in the neutral sentiment. Through this explanation, LIME provides insight into how the LSTM recognizes linguistic elements that lead it to classify the text as conveying a neutral sentiment.	

 

Figure 5.7: Neutral Text Classification 
and Explanation

	
Scenario 3: Negative Text:

The negative tweet on cryptocurrency from twitter was chosen and entered. Then the run model was clicked which transferred the text to the server-side model code for LSTM. The image was run in the model and it classified it as Fake. Then the explanations for the classification was done by the LIME (Local Interpretable Model-agnostic Explanations) package. 

Using its explanation methodology, LIME clarifies the neutral text classification in LSTM tweet classification. LIME accomplishes this by generating altered versions of the input tweet and examining how these affect the predictions made by the LSTM. These altered instances are then used to construct a surrogate model that attempts to simulate the behavior of the LSTM. LIME determines the crucial words and phrases that are most responsible for the neutral classification of a text. These key terms are highlighted along with significance scores in the explanation. If terms like "average," "standard," or similar expressions score higher, it suggests that they had a big part in the neutral sentiment. Through this explanation, LIME provides insights into how the LSTM deciphers linguistic elements, enabling it to classify the text as conveying a negative sentiment.

 

Figure 5.8: Negative Text Classification 
and Explanation


















6. Conclusion


6.1	Conclusion
We can conclude our findings and results as below.

•	Which CNN architecture should I use?
We can see that most of the classification using pre-trained CNN models such as ResNet, MobileNet, DenseNet, VGG16, etc. gave better results than custom-built CNN model. All these models have different and deep architecture. On a dataset with images of dimensions more than 100x100x3, these models would be an obvious choice. And, the data used for this project was with the dimensions 128x128x3. The VGG16 model gave the best results among the above-mentioned models for this data. 

•	How many layers should be included in the model?
From the results we can see that the model with just one layer outperformed the other models. There was not a significant difference in the performance of all those models. Another observation was that as I increased the complexity of the model by adding new layers or increasing the filter size, the performance deteriorated a little. When working with this dataset, using the basic CNN model is a good choice.

•	How to select the optimal hyperparameters?
While working with this project I experimented with changing the optimizer from Adam to SGD once. The performance of the model went down drastically. Using Adam optimizer would definitely be a better choice. We can also see that the models with dropout 0.25 perform better than the models with dropout 0.3. The selection of optimal hyperparameters can be done by trial-and-error method. There is a possibility that other set of hyperparameters would give better results.

•	How should we handle data?
The dataset used in this project is highly unbalanced. Some of the classes have much more instances than the others. This greatly affects the performance of the model. The model becomes sensitive to the classes with the greatest number of instances and classifies instances of other classes as that class. This huge imbalance in the medical dataset is common. It is very important to handle this by balancing the classes using methods such as data augmentation.

• Which LSTM architecture is suitable for text classification?
The evaluation of various LSTM architectures showed that using a single-layer LSTM with dropout gave promising results. More complex architectures with additional layers or increased dropout rates did not significantly improve performance. A basic LSTM structure proved effective for the given text classification task.

• How to determine the optimal sequence length?
Analyzing different sequence lengths revealed that shorter sequences around 80 words performed well. Extremely short sequences led to loss of information, while very long ones increased computational demands without substantial gains in accuracy. Therefore, a sequence length of around 80 words appears to strike a good balance between information retention and computational efficiency.

• Which XAI (Explainable AI) method was employed?
LIME (Local Interpretable Model-agnostic Explanations) was employed for explaining the LSTM model's predictions. LIME perturbs input texts and builds a surrogate model to interpret the LSTM's decisions. It identifies influential words and phrases that contribute to the specific classification outcome. This approach helped gain insights into how the model categorized texts as positive, neutral, or negative, making the model's predictions more understandable and interpretable.

• How to handle imbalanced text classification data?
The text classification dataset often exhibits class imbalance, where some classes have significantly more samples than others. This imbalance can skew the model's performance. To address this, techniques like oversampling, under sampling, or using weighted loss functions can help balance the class distribution. By augmenting the underrepresented classes, the model becomes more robust and less biased toward the dominant classes, ultimately enhancing classification accuracy across all classes.

• What is the significance of word embeddings in LSTM-based text classification?
Word embeddings represent words in a continuous vector space, which is essential for LSTM-based text classification. Word embeddings record the semantic relationships between words in place of one-hot encoding, allowing LSTMs to comprehend contextual meaning. Word embeddings in the LSTM model let the network learn from the order of words and their relationships, picking up on details like synonyms and sentiment. This makes it easier for the model to generalize and spot trends, which increases its efficiency in deciphering the underlying sentiment or context of a tweet. Word embeddings simply provide words a condensed and meaningful form, which helps the LSTM better understand linguistic intricacies and increase text categorization accuracy.


6.2	Future Work
The future of Explainable AI (XAI) in deep learning holds significant promise. In this process, interpretable neural networks are built, attention mechanisms are visualized, and predictions are associated with certain properties. Explanations will be more robust when ensembles, adversarial insights, and domain adaptability are used. Dynamic explanations, interaction between humans and AI, and ethical adherence are essential. Deep learning and symbolic AI hybrid models will provide performance and clarity. The rating of explanation quality will be consistent thanks to standardization. Prioritizing XAI will promote model openness, accountability, and user trust across a range of applications as deep learning progresses.

Future possibilities for Explainable AI (XAI) in fake image classification include attention processes, integrating models, counterfactual insights, and finer-grained explanations. Domain adaptability, adversarial and user-centered justifications, and objective dataset analysis are crucial. Collaboration between humans and AI must take ethical factors into account. As fake image production develops, uncertainty estimation and longitudinal analysis are essential. These developments aim to address new issues in fraudulent picture detection while enhancing transparency, accuracy, and consumer trust, the classifier models will be trained better.

Deep learning-based text sentiment analysis will lead to increased precision and context awareness. Models like RNNs and transformers, which take conversational context and multilingual subtleties into account, will increase sophisticated sentiment interpretation. The integration of emotion detection, domain adaption, and ethical considerations will all be crucial. The use of models will facilitate human-AI collaboration, clarify explanations, and assess temporal sentiment trends. These innovations will provide insightful information for corporations, social study, and understanding public mood.

Future scope for this paper:

The components of the application are decoupled and modular, and can be seamlessly integrated and plugged in to different application. One of the possible works is to create a chip which can handle the CNN or LSTM model running part, a concept known as "hardware acceleration." and then plug in the remaining code to this is circuit. This will reduce the time required for the computations as the hardware component will be very fast and the remaining components of the code are not too time consuming.

It is advantageous to implement the CNN and LSTM parts of the algorithm as dedicated hardware chips, or hardware acceleration. Specifically, this entails creating hardware circuits for CNN for image processing and LSTM for sequential data or text analysis. It is useful for applications like autonomous vehicles and robots because of its advantages, which include increased speed and efficiency thanks to parallel processing, decreased power consumption, real-time capabilities, and reduced latency. Performance can be improved by designing hardware chips specifically for a model.



 

Figure 6.1: Concept Design


Hardware acceleration does, however, provide certain challenges. Hardware chip design includes complex fabrication procedures and expertise. Although specialized hardware is excellent at certain jobs, it may not be as flexible as software. Costs related to chip design, production, and integration must also be taken into account. Compared to software systems, hardware systems are less adaptable to updates, and interface changes may be necessary for integration into current systems. In the end, the choice to use hardware acceleration should take into account the needs of the use case, financial limitations, ongoing maintenance requirements, and potential performance advantages that result from effective hardware implementations.






7.	Bibliography


1. Bird, J. J., & Lotfi, A. (2023). CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images. arXiv:2303.14126.

2. Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). Classification and Regression Trees. CRC Press.

3. Dhurandhar, A., Chen, P. Y., Luss, R., & Krause, A. (2018). Explanations Based on the Missing: Towards Contrastive Explanations with Pertinent Negatives. International Conference on Machine Learning.

4. Guidotti, R., Monreale, A., Ruggieri, S., Turini, F., Giannotti, F., & Pedreschi, D. (2018). A survey of methods for explaining black box models. ACM computing surveys.

5. Hinton, G. E., Osindero, S., & Teh, Y. W. (2006). A fast-learning algorithm for deep belief nets. Neural Computation.

6. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. Advances in neural information processing systems.

7. Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images.

8. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE.

9. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. Advances in neural information processing systems.

10. Mitchell, T. M. (1997). Machine Learning. McGraw Hill.

11. Nilsson, N. J. (1998). Artificial Intelligence: A New Synthesis. Morgan Kaufmann.

12. Ottavio, Calzone (2022). An intuitive explanation of LSTM.

13. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

14. Shortliffe, E. H. (2014). Computer-Based Medical Consultations: MYCIN. Elsevier.

15. Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. Advances in neural information processing systems.

16. Suran (2019). https://www.kaggle.com/datasets/skularat/bitcoin-tweets

17. Vapnik, V. N. (1995). The Nature of Statistical Learning Theory. Springer.

18. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems.

19. How to teach a computer to see with Convolutional Neural Networks [2022].
Image Classification with EfficientNet: Better performance with computational efficiency.

