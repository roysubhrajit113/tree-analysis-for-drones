#====================================
#------------- README ---------------
#====================================

# These are the tunable parameters according to your requirements, based on what
# drone is being used, and how close up shots they are.

# However, the below default parameters are selected in order to give an average
# view of the estimation of these tree attributes.
# You may use these, but for large-scale deployments, you may need to change each of them
# according to your requirements.

#====================================
#------------------------------------
#====================================

# HSV ranges for green detection
GREEN_LOWER = (35, 40, 40)
GREEN_UPPER = (85, 255, 255)

# Health detection threshold
HEALTHY_GREEN_RATIO = 0.05   # > 0.05 → Healthy

# Leaf presence threshold (number of green pixels)
LEAF_PIXEL_THRESHOLD = 1000

# Height categorization ratios
SMALL_TREE_RATIO = 0.2
MEDIUM_TREE_RATIO = 0.5

# Isolation threshold (in pixels)
ISOLATION_DISTANCE = 100

# Visible crown green-pixel threshold
VISIBLE_CROWN_THRESHOLD = 500

# Canopy coverage green ratio thresholds
CANOPY_SPARSE = 0.2
CANOPY_PARTIAL = 0.6
