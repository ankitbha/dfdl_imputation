package eagrn.fitnessfunction.impl.quality.impl;

import java.util.Map;

import eagrn.fitnessfunction.impl.quality.Quality;
import eagrn.fitnessfunction.impl.quality.TrendMeasureEnum;


/**
 * Try to minimize the quantity of high quality links (getting as close as possible
 * to 10 percent of the total possible links in the network) and at the same time maximize
 * the quality of these good links (maximize the mean of their confidence and weight adjustment).
 *
 * High quality links are those whose confidence-distance mean is above average.
 */

public class QualityMedianAboveAverage extends Quality {
    
    public QualityMedianAboveAverage(Map<String, Float[]> inferredNetworks) {
        super(inferredNetworks, TrendMeasureEnum.MEDIAN);
    }

    public double run(Map<String, Float> consensus, Double[] x) {
        /** 
         * 1. We create the map with the distances corresponding to the difference between the maximum and minimum 
         * of the vectors containing the mean between the weight and the distance normalized to the median of the total 
         * number of techniques 
         */
        Map<String, Float> distances = super.makeDistanceMap(consensus, x);

        /** 2. Calculate the mean of the confidence-distance means. */
        float conf, dist, confDistSum = 0;
        for (Map.Entry<String, Float> pair : consensus.entrySet()) {
            conf = pair.getValue();
            dist = distances.get(pair.getKey());
            confDistSum += (float) ((conf + (1 - dist)) / 2.0);
        }
        double mean = confDistSum / consensus.size();

        /** 3. Calculate the average of confidence-distance means of high quality links */
        confDistSum = 0;
        float confDist, cnt = 0;
        for (Map.Entry<String, Float> pair : consensus.entrySet()) {
            conf = pair.getValue();
            dist = distances.get(pair.getKey());
            confDist = (float) ((conf + (1 - dist)) / 2.0);
            if (confDist > mean) {
                confDistSum += confDist;
                cnt += 1;
            }
        }

        /** 4. Return fitness */
        return 1.0 - confDistSum/cnt;
    }
}
