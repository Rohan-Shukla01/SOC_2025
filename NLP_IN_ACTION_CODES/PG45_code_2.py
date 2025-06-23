import re
r=r"[^a-z]*(yo|[h']?ello|hey|hi|ok|(good[ ])?(morn[ing']{0,3}|afternoon|even[ing']{0,3}))[\s',.:;]*([a-z]{1,20})"
re_greet=re.compile(r,flags=re.IGNORECASE)
#s="Hello Rosa! Good morning on behalf of Victor"
#s="Good Morning Rosa Parks"
s="Good Morn'n Rosa Parks"
for match in re.finditer(re_greet,s):
    f=match.group
    print(f"group wise matches: \n main:{f(0)} \n g1 : {f(1)} \n g2: {f(2)} \n g3: {f(3)} \n g4: {f(4)}")