
prompt_person_summary = '''
Given the following observation about an individual {Name}, please summarize the relevant details from the profile and deduce their stance on {target}. His or her profile information is as follows:.
Name: {Name} | Location: {Location} | Description: {Description}
Stats: {Followers Count} followers, {Following Count} following, {Tweet Count} tweets
Account Created: {Account Created}
List of interest: {Related List}
Sample of posted tweets: {Sample of Previous Posts}
Task:
1. Summary: You can deduce the preferences and personality from profile, lists, and tweets, but please avoid repeating the observation in the summary.
2. Stance: Deduce the stance on {target}. Choose from [Support, Neutral, Oppose, Unknown].
Constraint: Only output the following format. No extra words.
Summary: [You are {Name}. followed by the narrative description]
Stance: [Your choice]
'''

prompt_person='''You are a Twitter user {username}. Given your personal information and recent memory, along with the current context and options, you are required to predict your most likely action:
(1) Your description: {role_description}; 
(2) Your stance on {target}: {stance};
(3) Current time is {current_time};
(4) The news page you got is {trigger_news}  
(5) Your recent memory is {past_event}
(6) The twitter page you can see is {tweet_page}
Determine your next action by selecting one option from each dimension:
(7) Action Type: select from [post, retweet, reply, quote]
(8) Action Object: "-" ONLY if type is 'post'; otherwise,  you must select the specific tweet ID from tweet_page that you are interacting with. (tweet_A or tweet_B or tweet_C)
(9) Action Content: "-" ONLY if type is 'retweet'; otherwise, you must select one item from {tweet_content}. (Do not generate your own or copy from elsewhere)
'''



MindStep = '''To predict your most likely action, you will simulate your cognitive processing step by step. Please follow this format strictly, No extra words:
My cognitive trace:
[Evocation-1] [Step-by-step summarize your persona and inherent behavioral patterns purely from your description, stance, and recent memory. Focus on your behavioral habits, identity, expression, and inclination towards posting original content or interacting with others.]
[Evocation-2] ...
...
[Percept-1] [Guided by your evoked persona and purpose, step-by-step summarize the key events from the current twitter page and news page that resonate with your persona, including both potential topics for independent posting and existing tweets for interaction.]
[Percept-2] ...
...
[Reaction-1] [Synthesizing your evocations and percepts, predict the specific action you would most likely take by prioritizing your persona and inherent behavioral habits over literal topic matching or merely reacting to visible tweets.]
[Reaction-...] ...
...
Therefore my action is:
Action type:
Action object:
Action content:
'''



#baseline
COT='''To predict your most likely action, you need to think step by step. Please follow this format strictly, No extra words: 
My thinking process:
the thinking process for choosing the action
Therefore my action is:
Action type:
Action object:
Action content:
'''

Direct_output='''Only output the following format, No extra words:
Action type:
Action object:
Action content:
'''
Plan_and_solve='''To predict your most likely action, first understand the problem, extract relevant context and available options, and devise a plan to solve the problem. Then, carry out the plan and solve the problem step by step. Please follow this format strictly, No extra words: 
Given:
the process for understand the problem 
Plan:
the process for devising a plan
Solve:
the process for carrying out the plan
Therefore my action is:
Action type:
Action object:
Action content:
'''

Hi_COT="""To predict your most likely action, you need to reason by alternating between <|instruction|> (planning what to do next) and <|execution|> (executing that plan). Each <|instruction|> describes the reasoning step or plan, and each <|execution|> performs the corresponding reasoning or computation. You should start with an instruction and follow this format strictly, No extra words:
<|instruction|> Step 1: ...
<|execution|> Step 1: ...
<|instruction|> Step 2: ...
<|execution|> Step 2: ...
...
Therefore my action is:
Action type: 
Action object: 
Action content:
"""


Persona_COT = '''To predict your most likely action, you need to think step by step, pay attention to your persona and historical behavioral tendencies, including your identity, behavioral habits, expression style, and preferences for posting or interacting with others. Your reasoning should reflect your personal behavioral tendencies rather than general patterns of how people typically act in similar situations. Please follow this format strictly, No extra words:
My thinking process:
the thinking process for choosing the action
Therefore my action is:
Action type:
Action object:
Action content:
'''

