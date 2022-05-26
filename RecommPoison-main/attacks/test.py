import os
isExists = os.path.exists('/root/autodl-tmp/RecommPoison-main/attacks/posion')
if not isExists:
    os.mkdir('/root/autodl-tmp/RecommPoison-main/attacks/model')
else:
    print("aahph")