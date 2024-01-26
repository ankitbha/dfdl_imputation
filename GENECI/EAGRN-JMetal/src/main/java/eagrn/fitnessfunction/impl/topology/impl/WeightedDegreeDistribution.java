package eagrn.fitnessfunction.impl.topology.impl;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import eagrn.fitnessfunction.impl.topology.Topology;

public class WeightedDegreeDistribution extends Topology {
    protected ArrayList<String> geneNames;

    public WeightedDegreeDistribution(ArrayList<String> geneNames) {
        this.geneNames = geneNames;
    }

    public double run(Map<String, Float> consensus, Double[] x) {
        
        Map<String, Float> undirectedDegreesMap = new HashMap<>();
        for (Map.Entry<String, Float> pair : consensus.entrySet()) {
            String[] genes = pair.getKey().split(";");
            undirectedDegreesMap.put(genes[0], undirectedDegreesMap.getOrDefault(genes[0], 0.0f) + pair.getValue());
            undirectedDegreesMap.put(genes[1], undirectedDegreesMap.getOrDefault(genes[1], 0.0f) + pair.getValue());
        }

        double[] undirectedDegreesPlusOne = new double[geneNames.size()];
        for (int i = 0; i < geneNames.size(); i++) {
            undirectedDegreesPlusOne[i] = undirectedDegreesMap.getOrDefault(geneNames.get(i), 0.0f) + 1.0;
        }
        return super.paretoTest(undirectedDegreesPlusOne);
    }
}
