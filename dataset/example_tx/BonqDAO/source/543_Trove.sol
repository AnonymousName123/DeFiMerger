//SPDX-License-Identifier: MIT
pragma solidity ^0.8.4;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "./IOwnable.sol";
import "./ITokenPriceFeed.sol";
import "./IMintableToken.sol";
import "./IMintableTokenOwner.sol";
import "./IFeeRecipient.sol";
import "./ILiquidationPool.sol";
import "./IStabilityPool.sol";
import "./ITrove.sol";

interface ITroveFactory {
  /* view */
  function lastTrove(address _trove) external view returns (address);

  function firstTrove(address _trove) external view returns (address);

  function nextTrove(address _token, address _trove) external view returns (address);

  function prevTrove(address _token, address _trove) external view returns (address);

  function containsTrove(address _token, address _trove) external view returns (bool);

  function stableCoin() external view returns (IMintableToken);

  function tokenOwner() external view returns (IMintableTokenOwner);

  function tokenToPriceFeed() external view returns (ITokenPriceFeed);

  function feeRecipient() external view returns (IFeeRecipient);

  function troveCount(address _token) external view returns (uint256);

  function totalDebt() external view returns (uint256);

  function totalCollateral(address _token) external view returns (uint256);

  function totalDebtForToken(address _token) external view returns (uint256);

  function liquidationPool(address _token) external view returns (ILiquidationPool);

  function stabilityPool() external view returns (IStabilityPool);

  function arbitragePool() external view returns (address);

  function getRedemptionFeeRatio(address _trove) external view returns (uint256);

  function getRedemptionFee(uint256 _feeRatio, uint256 _amount) external pure returns (uint256);

  function getBorrowingFee(uint256 _amount) external view returns (uint256);

  /* state changes*/
  function createTrove(address _token) external returns (ITrove trove);

  function createTroveAndBorrow(
    address _token,
    uint256 _collateralAmount,
    address _recipient,
    uint256 _borrowAmount,
    address _nextTrove
  ) external;

  function removeTrove(address _token, address _trove) external;

  function insertTrove(address _trove, address _newNextTrove) external;

  function updateTotalCollateral(
    address _token,
    uint256 _amount,
    bool _increase
  ) external;

  function updateTotalDebt(uint256 _amount, bool _borrow) external;

  function setStabilityPool(address _stabilityPool) external;

  function setArbitragePool(address _arbitragePool) external;

  // solhint-disable-next-line var-name-mixedcase
  function setWETH(address _WETH, address _liquidationPool) external;

  function increaseCollateralNative(address _trove, address _newNextTrove) external payable;

  /* utils */
  function emitLiquidationEvent(
    address _token,
    address _trove,
    address stabilityPoolLiquidation,
    uint256 collateral
  ) external;

  function emitTroveCollateralUpdate(
    address _token,
    uint256 _newAmount,
    uint256 _newCollateralization
  ) external;

  function emitTroveDebtUpdate(
    address _token,
    uint256 _newAmount,
    uint256 _newCollateralization,
    uint256 _feePaid
  ) external;
}
