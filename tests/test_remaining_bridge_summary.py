import unittest
from scripts.vision.finalize_remaining_bridge_review import summarize
from scripts.vision.remaining_bridge_observations import expanded

class SummaryTests(unittest.TestCase):
    def fixture(self):
        frames=[];items=[]
        for index in range(3):
            mid=str(index);objects=[dict(instance_label=n,device_id=str(n),category='switchgear') for n in range(53)]
            frames.append(dict(member_id=mid,category='switchgear',objects=objects))
            for obj in objects:items.append(dict(member_id=mid,review_id=f'{mid}-{obj["instance_label"]}',runtime_label=obj['instance_label'],object_id=obj['device_id'],review_status='实例可见且内容可辨识'))
        return items,dict(frames=frames,groups={'lineage':['0','1','2']})

    def test_no_automatic_training_admission(self):
        items,trace=self.fixture();group=summarize(items,trace)['lineage']
        self.assertFalse(group['training_admitted']);self.assertFalse(group['promotable'])
        self.assertEqual(group['status'],'visibility_checked_only_not_training_admitted')

    def test_one_risk_holds_whole_group(self):
        items,trace=self.fixture();items[0]['review_status']='实例有可见证据，但内容不足'
        self.assertEqual(summarize(items,trace)['lineage']['status'],'held_for_content_risk')

    def test_missing_duplicate_and_mapping_conflict(self):
        items,trace=self.fixture()
        with self.assertRaises(ValueError):summarize(items[:-1],trace)
        items[-1]=items[0]
        with self.assertRaises(ValueError):summarize(items,trace)
        items,trace=self.fixture();items[0]['object_id']='wrong'
        with self.assertRaises(ValueError):summarize(items,trace)

    def test_explicit_review_coverage(self):
        self.assertEqual(set(expanded()),{f'M{i:03}' for i in range(1,136)})

if __name__=='__main__':unittest.main()
