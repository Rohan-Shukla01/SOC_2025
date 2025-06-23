import re 
r=re.compile(r"(hi|hello|hey)[ ]*\w*",flags=re.IGNORECASE)
for match in re.finditer(r,"Hello Rosa!! I wanted to say a hi."):
    #flags=re.IGNORECASE should be passed at the compile time only
    print(f"Full match: {match.group(0)}")
    print(f"Group match: {match.group(1)}")