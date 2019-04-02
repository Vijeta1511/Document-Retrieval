#Vijeta Agrawal 
#Registration No. 180127667
import numpy as np
import math

class Retrieve:
    
    # Create new Retrieve object storing index and termWeighting scheme
    def __init__(self,index, termWeighting):
        self.index = index
       
         #for binary
        self.DocumentTermDict_Binary = {} #to store term weighting Scheme
        self.termWeighting = termWeighting
        for i in self.index:
            for j in self.index[i]:
                if j not in self.DocumentTermDict_Binary:
                    self.DocumentTermDict_Binary[j] = 1
                else:
                    self.DocumentTermDict_Binary[j] = self.DocumentTermDict_Binary[j] + 1
        
        #for TF
        self.DocumentTermDict_tf = {} #to store term weighting Scheme
        self.termWeighting = termWeighting
        for i in self.index:
            for j in self.index[i]:
                if j not in self.DocumentTermDict_tf:
                    self.DocumentTermDict_tf[j] = self.index[i][j]**2
                else:
                    self.DocumentTermDict_tf[j] = self.DocumentTermDict_tf[j] + self.index[i][j]**2
        
        #calculating inverted document frequency
        self.lengthofdoc = len(self.DocumentTermDict_Binary)
        self.inverted_doc_freq = {}
        for i in self.index:
            idf = len(self.index[i])
            idf = self.lengthofdoc/idf
            idf = math.log(idf,10)
            self.inverted_doc_freq[i] = idf
        
        #for TF.IDF
        self.DocumentTermDict_tfidf = {} #to store term weighting Scheme
        self.termWeighting = termWeighting
        for i in self.index:
            for j in self.index[i]:
                if j not in self.DocumentTermDict_tfidf:
                    self.DocumentTermDict_tfidf[j] = (self.index[i][j]*self.inverted_doc_freq[i])**2
                else:
                    self.DocumentTermDict_tfidf[j] = self.DocumentTermDict_tfidf[j] + (self.index[i][j]*self.inverted_doc_freq[i])**2
          
# Creating candidate List for Retrieval Process
    def createCandidateList(self,query):
        CandidateList = []
        for i in query:
            try:
                for l in self.index[i]:
                    CandidateList.append(l)
            except:
                continue
        CandidateList = list(np.unique(CandidateList)) 
        return CandidateList
    
 # Method performing retrieval for specified query
    def forQuery(self, query):
        CandidateList = self.createCandidateList(query)
        if self.termWeighting=="binary":
            return self.BinaryTermWeighting(query,CandidateList)
        elif self.termWeighting=="tf":
            return self.Tf_TermWeighting(query,CandidateList)
        else :
            return self.Tfidf_TermWeighting(query,CandidateList)  #else tfidf scheme  

 
 #calculating Binary term weighting as well as Document Query Similarity in WeightandSimilarity and storing in BinaryDict[]
    def BinaryTermWeighting(self,query,CandidateList):
        BinaryDict = {} 
        for j in CandidateList:
            WeightandSimilarity = 0
            for i in query:
                try:
                    if j in self.index[i]:
                        WeightandSimilarity = WeightandSimilarity + 1
                    
                except:
                    continue #Calculating Document Query Similarity using Vector Space Model
            WeightandSimilarity = WeightandSimilarity/np.sqrt(self.DocumentTermDict_Binary[j])
            BinaryDict[j] = WeightandSimilarity
        #Sorting BinaryDict to rank the documents    
        sortedBinaryDict = sorted(BinaryDict,key = BinaryDict.__getitem__,reverse = True)[:10]#storing top 10 relevant docs
        print(sortedBinaryDict)
        return (sortedBinaryDict)
 
 #calculating Tf term weighting as well as Document Query Similarity in WeightandSimilarity and storing in Tf_Dict[]
    def Tf_TermWeighting(self,query,CandidateList):
        Tf_Dict = {} 
        for j in CandidateList:
            WeightandSimilarity = 0
            for i in query:
                try:
                    if j in self.index[i]:
                        WeightandSimilarity = WeightandSimilarity + self.index[i][j]
                    
                except:
                    continue #Calculating Document Query Similarity using Vector Space Model
            WeightandSimilarity = WeightandSimilarity/np.sqrt(self.DocumentTermDict_tf[j])
            Tf_Dict[j] = WeightandSimilarity
        #Sorting Tf_Dict to rank the documents    
        sortedTf_Dict = sorted(Tf_Dict,key = Tf_Dict.__getitem__,reverse = True)[:10]#storing top 10 relevant docs
        print(sortedTf_Dict)
        return (sortedTf_Dict)
    
 #calculating Tfidf term weighting as well as Document Query Similarity in WeightandSimilarity and storing in Tfidf_Dict[]
    def Tfidf_TermWeighting(self,query,CandidateList):
        Tfidf_Dict = {} 
        for j in CandidateList:
            WeightandSimilarity = 0
            for i in query:
                try:
                    if j in self.index[i]:
                        WeightandSimilarity = WeightandSimilarity + self.index[i][j]*query[i]*(self.inverted_doc_freq[i]**2)
                    
                except:
                    continue #Calculating Document Query Similarity using Vector Space Model
            WeightandSimilarity = WeightandSimilarity/np.sqrt(self.DocumentTermDict_tfidf[j])
            Tfidf_Dict[j] = WeightandSimilarity
         #Sorting Tfidf_Dict to rank the documents    
        sortedTfidf_Dict = sorted(Tfidf_Dict,key = Tfidf_Dict.__getitem__,reverse = True)[:10] #storing top 10 relevant docs
        print(sortedTfidf_Dict)
        return (sortedTfidf_Dict)