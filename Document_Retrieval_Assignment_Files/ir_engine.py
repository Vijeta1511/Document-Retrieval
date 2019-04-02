"""\
------------------------------------------------------------
USE: python <PROGNAME> (options)
OPTIONS:
    -unigram : print this help message
    -bigram : use "with stoplist" configuration (default: without)
    -trigram : use "with stemming" configuration (default: without)
------------------------------------------------------------\
"""

#==============================================================================
# Importing

import sys, getopt, re, time
from my_retriever import Retrieve

#==============================================================================
# Command line processing

class CommandLine:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:], 'hspw:o:')
        opts = dict(opts)
        self.exit = True

        if '-h' in opts:
            self.printHelp()
            return

        if len(args) > 0:
            print("*** ERROR: no arg files - only options! ***", file=sys.stderr)
            self.printHelp()
            return

        if '-w' in opts:
            if opts['-w'] in ('binary', 'tf', 'tfidf'):
                self.termWeighting = opts['-w']
            else:
                warning = (
                    "*** ERROR: term weighting label (opt: -w LABEL)! ***\n"
                    "    -- value (%s) not recognised!\n"
                    "    -- must be one of: binary / tf / tfidf"
                    )  % (opts['-w'])
                print(warning, file=sys.stderr)
                self.printHelp()
                return
        else:
            self.termWeighting = 'binary'

        if '-o' in opts:
            self.outfile = opts['-o']
        else:
            print("*** ERROR: must specify output file (opt: -o FILE) ***",
                  file=sys.stderr)
            self.printHelp()
            return

        if '-s' in opts and '-p' in opts:
            self.indexFile   = 'index_withstoplist_withstemming.txt'
            self.queriesFile = 'queries_withstoplist_withstemming.txt'
        elif '-s' in opts:
            self.indexFile   = 'index_withstoplist_nostemming.txt'
            self.queriesFile = 'queries_withstoplist_nostemming.txt'
        elif '-p' in opts:
            self.indexFile   = 'index_nostoplist_withstemming.txt'
            self.queriesFile = 'queries_nostoplist_withstemming.txt'
        else:
            self.indexFile   = 'index_nostoplist_nostemming.txt'
            self.queriesFile = 'queries_nostoplist_nostemming.txt'
            
        self.exit = False

    def printHelp(self):
        progname = sys.argv[0]
        progname = progname.split('/')[-1] # strip off extended path
        help = __doc__.replace('<PROGNAME>', progname, 1)
        print(help, file=sys.stderr)

#==============================================================================
# Timer 

class MyTimer:
    def __init__(self):
        self.startTime = {}

    def start(self, label = None):
        self.startTime[label] = time.clock()

    def stopPrint(self, label = None):
        duration = time.clock() - self.startTime[label]
        msg = 'TIME (%s): %.2f' % (label, duration)
        print(msg, file=sys.stderr)

#==============================================================================
# Load (precomputed) Index File for (preprocessed) Document Collection

class IndexLoader:
    def __init__(self, indexFile):
        self.index = {}
        docidCountRE = re.compile('(\d+):(\d+)')
        f = open(indexFile, 'r')
        for line in f:
            term = line.split(' ', 1)[0]
            self.index[term] = {}
            for (docid, count) in docidCountRE.findall(line):
                docid = int(docid)
                self.index[term][docid] = int(count)

    def getIndex(self):
        return self.index

#==============================================================================
# Load (preprocessed) Collection of Queries

class Queries:
    def __init__(self, queriesFile):
        self.qStore = {}
        termCountRE = re.compile('(\w+):(\d+)')
        f = open(queriesFile, 'r')
        for line in f:
            qid = int(line.split(' ', 1)[0])
            self.qStore[qid] = {}
            for (term, count) in termCountRE.findall(line):
                self.qStore[qid][term] = int(count)
    
    def getQuery(self, qid):
        if qid in self.qStore:
            return self.qStore[qid]
        else:
            print("*** ERROR: unknown query identifier (\"%s\") ***" % qid, file=sys.stderr)
            if type(qid) == type(''):
                print('WARNING: query identifiers should be of type: integer', file=sys.stderr)
                print('         -- your query identifier is of type: string', file=sys.stderr)
            print(' -- program exiting', file=sys.stderr)
    
    def qids(self):
        return sorted(self.qStore)

#==============================================================================
# Store for Retrieval Results

class ResultStore:
    def __init__(self, outfile):
        self.outfile = outfile
        self.results = []

    def store(self, qid, docids):
        if len(docids) > 10:
            docids = docids[:10]
        self.results.append((qid, docids))

    def output(self):
        with open(self.outfile, 'w') as out:
            for (qid, docids) in self.results:
                for docid in docids:
                    print(qid, docid, file=out)

#==============================================================================
# MAIN

if __name__ == '__main__':

    config = CommandLine()
    if config.exit:
        sys.exit(0)        
    index = IndexLoader(config.indexFile).getIndex()
    retrieve = Retrieve(index, config.termWeighting)
    queries = Queries(config.queriesFile)
    allResults = ResultStore(config.outfile)

    t = MyTimer()
    t.start('retrieval')

    for qid in queries.qids():
        query = queries.getQuery(qid)
        results = retrieve.forQuery(query)
        allResults.store(qid, results)

    t.stopPrint('retrieval')    
    allResults.output()

