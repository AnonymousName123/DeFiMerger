//SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.4;

import "./trove-factory.sol";

contract OriginalTroveFactory is TroveFactory {
  function name() public view override returns (string memory) {
    return "Factory V1.1.0";
  }

  function redeemStableCoinForCollateral(address _collateralToken, uint256 _stableAmount, uint256 _maxRate, uint256 _lastTroveCurrentICR, address _lastTroveNewPositionHint)
    public
    override
  {
    require(_collateralToken != 0x388E289A1705fa7b8808AB13f0e0f865E2Ff94eE && _collateralToken != 0xA1Dd21527c76BB1A3b667149e741A8B0f445FaE2, "LP tokens can not be redeemed");
    super.redeemStableCoinForCollateral(_collateralToken, _stableAmount, _maxRate, _lastTroveCurrentICR, _lastTroveNewPositionHint);
  }
}