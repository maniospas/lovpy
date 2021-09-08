class ClusterNode:
    def perform_maintainance(self):
        pass

    def checkpoint(self):
        pass

    def receive_big_data(self):
        pass

    def request_data_processing(self):
        pass

    def visualize(self):
        pass

    def offload(self):
        pass


node = ClusterNode()
node.perform_maintainance()
node.receive_big_data()
node.checkpoint()
node.request_data_processing()
node.visualize()
print("Failed to catch error.")
node.offload()
