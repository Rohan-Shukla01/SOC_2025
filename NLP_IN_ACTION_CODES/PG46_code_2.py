import re
r=r"[^a-z]*(yo|[h']?ello|hey|hi|ok|(good[ ])?(morn[ing']{0,3}|afternoon|even[ing']{0,3}))[\s',.:;]*([a-z]{1,20})"
re_greet=re.compile(r,flags=re.IGNORECASE)
my_names=set(['rosa','rose','bot','chatty','chatterbot','chatbot'])
#names by which the chatbot expects the user to call it
curt_names=set(['hal','you','u'])
#names used in a sarcastic way by user, ref to some old show
greeter_name='' #name of the user, we dont care about it in this specific code
match=re.search(re_greet,input())
if match:
    at_name=match.groups()[-1]
    if at_name in curt_names:
        print("Good One...")
    elif at_name in my_names:
        print(f"HI {greeter_name}, How are you?")