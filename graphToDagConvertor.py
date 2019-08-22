import argparse
import networkx as nx

def graphConstructor(inputFileName):
    edgeList = []
    with open(inputFileName, 'r') as fin:
        for line in fin:
            line = line.split('\t')
            line[-1] = line[-1][:-1]
            node1 = line[0]
            node2 = line[1]
            weight = float(line[2]) 
            edgeList.append((node1, node2, weight))

    # Constructing a weighted graph 
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(edgeList)
    return graph 

def graphToDagConvertor(graph):
    # Divide graph into several strongly connected components
    connectedComponents = list(nx.strongly_connected_components(graph))
    connectedComponents = [component for component in connectedComponents if len(component) != 1]

    # Iterate over connected components of graph and break all cycles
    for component in connectedComponents:
        subGraph = graph.subgraph(component)
        subGraph = nx.DiGraph(subGraph)
        while(1):
            try:
                cycle = nx.find_cycle(subGraph)
            except:
                break 
            # Remove edge with the smallest weight
            weight = [subGraph[x[0]][x[1]]['weight'] for x in cycle]
            minIndex = weight.index(min(weight))
            deleteEdge = cycle[minIndex]
            subGraph.remove_edge(deleteEdge[0], deleteEdge[1])
            graph.remove_edge(deleteEdge[0], deleteEdge[1])

    return graph

def dagSaver(dag, outputFileName):
    # Save the generated dag to disk 
    with open(outputFileName, 'w') as fout:
        for e in graph.edges():
                fout.write('{}\t{}\t{}\n'.format(e[0], e[1], graph[e[0]][e[1]]['weight']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Provide input and output address for graph and dag")
    parser.add_argument('inputFileName', help = 'address of graph file in the format of\
                                                    hypernym`\\t`hyponym`\\t`score')
    parser.add_argument('outputFileName', help = 'address of dag file in the format of\
                                                    hypernym`\\t`hyponym`\\t`score')
    args = parser.parse_args()
    graph = graphConstructor(args.inputFileName)
    dag = graphToDagConvertor(graph)
    dagSaver(dag, args.outputFileName)
