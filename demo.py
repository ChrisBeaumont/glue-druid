import numpy as np
import json

from glue.core import Data, BaseCartesianData, DataCollection, ComponentID
from glue.core.subset import RangeSubsetState
from glue.app.qt.application import GlueApplication
from glue.utils import view_shape
import requests


def hist(uri, data_source, dim, range, bins, filter_dim, filter_range):
    size = (range[1] - range[0]) / bins
    offset = range[0]

    query = {
      "queryType" : "groupBy",
      "dataSource" : data_source,
      "intervals" : ["2013-01-01/2013-04-01"],
      "granularity" : "all",
      "dimensions" :[
        {
            "type": "extraction",
            "dimension": dim,
            "outputName": "bin",
            "extractionFn": {"type": "bucket", "size": size, "offset": offset},
            "outputType": "FLOAT"
        }
      ],
      "filter": {
          "type": "and",
          "fields": [
            {
                "type": "bound",
                'dimension': dim,
                'lower': range[0],
                'upper': range[1],
                'ordering': 'numeric',
            }
          ]
      },
      "aggregations" : [
        {
          "type" : "count",
          "name" : "count"
        }
      ]
    }

    if filter_dim:
        query['filter']['fields'].append({
            "type": "bound",
            'dimension': filter_dim,
            'lower': filter_range[0],
            'upper': filter_range[1],
            'ordering': 'numeric'
        })

    print(json.dumps(query))

    result = requests.post(uri + '/druid/v2', json=query)
    result.raise_for_status()

    response = result.json()
    result = np.zeros(bins)
    for rec in response:
        x = rec['event']['bin']
        idx = int((x - offset) // size)
        if idx < 0 or idx >= len(result):
            continue
        result[idx] = rec['event']['count']

    return result



class DruidData(Data):

    def __init__(self, uri, datasource):
        self.uri = uri
        self.datasource = datasource
        self.cids = [ComponentID('passenger_count'), ComponentID('trip_distance'), ComponentID('trip_time_in_secs')]
        super(DruidData, self).__init__(label=self.datasource)

    @property
    def shape(self):
        return (int(1e9),)

    @property
    def main_components(self):
        return self.cids

    def get_kind(self, cid):
        return 'numerical'

    def get_data(self, cid, view=None):
        print(cid)
        return np.random.random(view_shape(self.shape, view))

    def get_mask(self, subset_state, view=None):
        return subset_state.to_mask(self, view=view)

    def compute_statistic(self, statistic, cid, axis=None, finite=True, positive=False, subset_state=None, percentile=None, random_subset=None):
        if axis is None:
            if statistic == 'minimum':
                return 0
            elif statistic == 'maximum':
                if cid in self.pixel_component_ids:
                    return self.shape[cid.axis]
                else:
                    return 1
            elif statistic == 'mean' or statistic == 'median':
                return 0.5
            elif statistic == 'percentile':
                return percentile / 100
            elif statistic == 'sum':
                return self.size / 2
        else:
            final_shape = tuple(self.shape[i] for i in range(self.ndim)
                                if i not in axis)
            return np.random.random(final_shape)


    def compute_histogram(self, cid, range=None, bins=None, log=False, subset_state=None, subset_group=None):
        print("hist", cid, range, bins, log, subset_state, subset_group)

        filter_dim = filter_range = None
        if isinstance(subset_state, RangeSubsetState):
            filter_dim = subset_state.att.label
            filter_range = subset_state.lo, subset_state.hi

        print(filter_dim, filter_range)
        return hist(
            self.uri,
            self.datasource,
            cid[0].label,
            range[0],
            bins[0],
            filter_dim,
            filter_range
        )


druid_uri = 'http://ec2-54-153-26-61.us-west-1.compute.amazonaws.com:8082'
d = DruidData(druid_uri, 'taxi')

print(d.compute_histogram(d.cids[0:1], range=[[0, 5]], bins=[5]))
dc = DataCollection([d])
ga = GlueApplication(dc)
ga.start()
