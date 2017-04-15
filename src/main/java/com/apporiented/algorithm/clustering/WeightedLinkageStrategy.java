package com.apporiented.algorithm.clustering;

import java.util.Collection;

public class WeightedLinkageStrategy implements LinkageStrategy {

    @Override
    public Distance calculateDistance(Collection<Distance> distances) {
        double sum = 0;
        double weightTotal = 0;
        for (Distance distance : distances) {
            weightTotal += distance.getWeight();
            sum += distance.getDistance() * distance.getWeight();
        }

		return new Distance(sum / weightTotal, weightTotal);
    }
}
