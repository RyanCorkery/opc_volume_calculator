from OPC_detector import OPCDetector


def main():
    opc_predictor = OPCDetector()
    volume = opc_predictor.get_volume_from_stack('opc_stack_2')
    print(volume)


if __name__ == '__main__':
    main()
