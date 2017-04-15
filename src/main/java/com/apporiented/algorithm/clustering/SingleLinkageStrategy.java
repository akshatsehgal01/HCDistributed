package com.apporiented.algorithm.clustering;

import java.util.Collection;

public class SingleLinkageStrategy implements LinkageStrategy {

	@Override
	public Distance calculateDistance(Collection<Distance> distances) {
		double min = Double.NaN;

		for (Distance dist : distances) {
		    if (Double.isNaN(min) || dist.getDistance() < min)
		        min = dist.getDistance();
		}
		return new Distance(min);
	}
}
