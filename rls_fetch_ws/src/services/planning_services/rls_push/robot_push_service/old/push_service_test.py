import rospy
from rls_push_msgs.srv import *

PUSH_SER_NAME = 'fetch_push'

if __name__=='__main__':
    rospy.init_node('test_push')
    rospy.wait_for_service(PUSH_SER_NAME)
    client = rospy.ServiceProxy(PUSH_SER_NAME, FetchPush)

    print 'service is listened'
#    raw_input('find marker?')
#    req = FetchPushRequest()
#    req.request = -1
#    client(req)

    #raw_input('push to left?')
    ## push to left
    #req = FetchPushRequest()
    #req.request = 1
    #req.distance = 0.2
    #client(req)

    raw_input('push to right?')
    # push to right
    req = FetchPushRequest()
    req.request = 2
    req.distance = 0.4
    client(req)

    #raw_input('push to edge?')
    ## push to edge
    #req = FetchPushRequest()
    #req.request = 0
    #client(req)


    rospy.spin()



