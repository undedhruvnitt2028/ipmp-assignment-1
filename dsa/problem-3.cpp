class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int theone = 0;
        for (int i = 0; i<nums.size(); i++)
        {
            int count = 0;
            
            for (int j = 0; j<nums.size(); j++)
            {
                if (nums[i] == nums[j])
                {
                    count += 1;
                }
                else
                {
                    continue;
                }
            }
            if (count == 1)
            {
                theone = nums[i];
                break;
            }
            else
            {
                continue;
            }
        }
        return theone;
    }
};