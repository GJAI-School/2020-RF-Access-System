def findDecision(obj): #obj[0]: distance
   if obj[0]>0.9296:
      return 'No'
   elif obj[0]<=0.9296:
      return 'Yes'
   else:
      return 'Yes'
