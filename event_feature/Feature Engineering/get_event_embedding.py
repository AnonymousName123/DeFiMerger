import os
import re
import time
import traceback
import numpy as np
import pandas as pd
import hashlib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

def price_manipulation_feature_engineering(record_tuple, historical_data=None):
    """
    Feature Engineering for Price Manipulation Attacks
    Input:
    record_tuple: (Number, Address, Name, Topics, Data)
    historical_data: Historical transaction data (optional, for time-series features)
    Output: High-dimensional feature vector related to price manipulation attacks
    """
    number, address, name, topics, data = record_tuple

    # ==================== Basic features====================
    features = {}

    # Sequence number/Temporal features
    features['number'] = int(number) if number != '' else 0
    features['number_norm'] = min(features['number'] / 1000, 1e5) 

    # Address features
    if address and address != '':
        # Basic hashing features
        addr_hash = hashlib.sha256(address.encode()).digest()
        features['addr_hash1'] = int.from_bytes(addr_hash[:4], 'big') / (2 ** 32 - 1)
        features['addr_hash2'] = int.from_bytes(addr_hash[4:8], 'big') / (2 ** 32 - 1)
        features['addr_length'] = len(address) / 100

        # Address features related to price manipulation
        features['is_contract_address'] = 1.0 if len(address) == 42 else 0.0
        features['is_honey_pot_address'] = 1.0 if address and any(
            pattern in address.lower() for pattern in ['0x000000', 'dead', '0000']) else 0.0
        features['addr_entropy'] = calculate_address_entropy(address)
    else:
        features['addr_hash1'] = 0.0
        features['addr_hash2'] = 0.0
        features['addr_length'] = 0.0
        features['is_contract_address'] = 0.0
        features['is_honey_pot_address'] = 0.0
        features['addr_entropy'] = 0.0

    # Event Name Features
    if name and name != '':
        # Core transaction types
        features['is_transfer'] = 1.0 if 'Transfer' in name else 0.0
        features['is_approval'] = 1.0 if 'Approval' in name else 0.0
        features['is_swap'] = 1.0 if any(keyword in name for keyword in ['Swap', 'Trade', 'Exchange']) else 0.0
        features['is_mint_burn'] = 1.0 if any(keyword in name for keyword in ['Mint', 'Burn']) else 0.0

        # Event type
        features['is_token_operations'] = 1.0 if any(keyword.lower() == name.lower for keyword in [
        "Approval",
        "ApprovalByPartition",
        "BalanceChange",
        "BalanceTransfer",
        "BalanceUpdated",
        "Burn",
        "Burned",
        "BurnedFromLiquidityPool",
        "Converted",
        "ConvertedToElastic",
        "ConvertedToRigid",
        "InitialMint",
        "issue",
        "Issued",
        "IssuedOTokens",
        "log",
        "LogMint",
        "LogTransfer",
        "Mint",
        "Minted",
        "MintOnDeposit",
        "MyTransfer",
        "RPLFixedSupplyBurn",
        "Supply",
        "TotalSupplyUpdated",
        "Transfer",
        "TransferBatch",
        "TransferByPartition",
        "TransferShares",
        "TransferSingle",
        "TokensMinted",
        "TokensTransferred",
        "ZeroBurn"
      ]) else 0.0
        features['is_liquidity'] = 1.0 if any(keyword.lower() == name.lower for keyword in [
        "AddLiquidity",
        "BinsSet",
        "Collect",
        "CollectFees",
        "CollectPermanentPoolAmount",
        "CollectSwapFees",
        "CompositionFees",
        "DecreasePoolAmount",
        "DepositedToBins",
        "DODOFlashLoan",
        "Flash",
        "FlashLoan",
        "IncreaseLiquidity",
        "IncreasePoolAmount",
        "LiquidityAdded",
        "LiquidityChanged",
        "LiquidityDeposited",
        "LiquidityRemoved",
        "LiquidityUpdated",
        "LiquidityWithdrawn",
        "LogFlashLoan",
        "LogJoin",
        "LogExit",
        "LogUpdatePool",
        "PairCreated",
        "PoolBalanceChanged",
        "PoolBalanced",
        "RemoveLiquidity",
        "RemoveLiquidityImbalance",
        "RemoveLiquidityOne",
        "Sync",
        "UpdatePositionLiquidity",
        "WithdrawnFromBins",
        "ActivePoolETHBalanceUpdated",
        "AutoNukeLP"
      ]) else 0.0
        features['is_trading'] = 1.0 if any(keyword.lower() == name.lower for keyword in [
        "AssetSwapped",
        "AtomicSynthExchange",
        "Buy",
        "BuyNF",
        "BuyToken",
        "BuyUSDG",
        "BuyUSDP",
        "ClosePosition",
        "DecreasePosition",
        "DebtSwap",
        "EthPurchase",
        "Exchange",
        "ExchangeEntryAppended",
        "ExchangeEntrySettled",
        "ExchangeTracking",
        "ExecuteDecreaseOrder",
        "ExecuteTrade",
        "IncreasePosition",
        "KyberTrade",
        "Sell",
        "SellGem",
        "SellNF",
        "SellOrderCompleted",
        "SellUSDG",
        "SellUSDP",
        "Swap",
        "Swap2PK",
        "SwapAndLiquify",
        "SwapOrderId",
        "Swapped",
        "SwappedFromVUsd",
        "SwappedToVUsd",
        "SwappedV3",
        "SwapToBan",
        "SwapTokensForETH",
        "SynthExchange",
        "TokenExchange",
        "TokenExchangeUnderlying",
        "TokenPurchase",
        "TokensBought",
        "TokensBoughtAndDeposited",
        "TokensSwapped",
        "TokenSwap",
        "TradeExecute",
        "UpdatePosition",
        "WooSwap"
      ]) else 0.0
        features['is_lending'] = 1.0 if any(keyword.lower() == name.lower for keyword in [
        "AccrueInterest",
        "AddInterest",
        "Borrow",
        "BorrowAllowanceDelegated",
        "Borrowed",
        "BorrowedAdded",
        "Borrowing",
        "CalculateBorrowRate",
        "CalculateKinkBorrowRate",
        "DebtUpdate",
        "DepositCollateral",
        "EnterMarket",
        "ERC20CollateralAdded",
        "ExitMarket",
        "Interest",
        "InterestAccrued",
        "InterestShortCircuit",
        "InterestStreamRedirected",
        "IsolationModeTotalDebtUpdated",
        "LiabilityAdded",
        "LiabilityRemoved",
        "LoanTaken",
        "LogAccrue",
        "LogBorrow",
        "MarketEntered",
        "NewBorrow",
        "NewCollateralFactor",
        "PayLendingFee",
        "PoolInterestUpdated",
        "Repay",
        "RepayBorrow",
        "Repayment",
        "RequestBorrow",
        "RequestRepay",
        "ReserveDataUpdated",
        "ReservesAdded",
        "ReservesUpdated",
        "ReserveUpdated",
        "ReserveUsedAsCollateralDisabled",
        "ReserveUsedAsCollateralEnabled",
        "RestructureBadDebt",
        "RestructureDebt",
        "SuppliedLink",
        "TroveCollateralUpdate",
        "TroveCreated",
        "TroveDebtUpdate",
        "TroveInserted",
        "TroveSnapshotsUpdated",
        "TroveUpdated",
        "UpdateInterest",
        "UserCollateralChanged",
        "WithdrawCollateral"
      ]) else 0.0
        features['is_debt'] = 1.0 if any(keyword.lower() == name.lower for keyword in [
        "AccountRemovedFromLiquidation",
        "ActivePoolLUSDDebtUpdated",
        "Liquidated",
        "LiquidationCall",
        "TroveLiquidated",
        "TroveRemoved",
        "DebtCacheUpdated"
      ]) else 0.0
        features['is_oracle'] = 1.0 if any(keyword.lower() == name.lower for keyword in [
        "AnchorPriceUpdated",
        "BaseRateUpdated",
        "CachedAssetPrice",
        "DailyDataUpdate",
        "LogExchangeRate",
        "NewReport",
        "PriceDataUpdate",
        "PriceRateCacheUpdated",
        "PriceRecorded",
        "PriceUpdated",
        "RateUpdated",
        "SetRate",
        "ShareRateChange",
        "TokenRateCacheUpdated",
        "TokenRateUpdate",
        "TwapUpdated",
        "TweakPrice",
        "UniswapWindowUpdated",
        "UpdateEMA",
        "UpdateExchangeRate",
        "UpdateFundingRate",
        "UpdateRate"
      ]) else 0.0
        features['is_staking'] = 1.0 if any(keyword.lower() == name.lower for keyword in [
        "ApplyReward",
        "ArriveFeeRewards",
        "AssetIndexUpdated",
        "Claim",
        "ClaimAdminFee",
        "ClaimRewards",
        "CollectMarginFees",
        "ConvertDustToEarned",
        "DepositRewards",
        "DistributedBorrowerComp",
        "DistributedBorrowerPBX",
        "DistributedBorrowerReward",
        "DistributedBorrowerVenus",
        "DistributedBorrowerXcn",
        "DistributedSupplierCan",
        "DistributedSupplierComp",
        "DistributedSupplierFortress",
        "DistributedSupplierPBX",
        "DistributedSupplierReward",
        "DistributedSupplierRifi",
        "DistributedSupplierVenus",
        "DistributedSupplierXcn",
        "DistributionAccumulatorIncreased",
        "Dividend",
        "DividendsDistributed",
        "DividendWithdrawn",
        "FeeDistributed",
        "GetReward",
        "Harvest",
        "Harvested",
        "LogOnReward",
        "PinkRewardPaid",
        "PinkPaid",
        "ProcessedDividendTracker",
        "ProfitPaid",
        "ReceiveReward",
        "Reward",
        "RewardAdded",
        "RewardDistributed",
        "Rewarded",
        "RewardPaid",
        "RewardsAccrued",
        "RewardsClaimed",
        "RewardTokenCollected",
        "Stake",
        "Staked",
        "StakeEnd",
        "StakeGlp",
        "StakeGoodAccounting",
        "StakePlp",
        "StakeStart",
        "StakeWithdraw",
        "StakingDecreased",
        "StakingIncreased",
        "TotalStakesUpdated",
        "Unstaked",
        "UnstakeGlp",
        "UnstakePlp",
        "YieldDistribution",
        "Reinvest"
      ]) else 0.0
        features['is_vaults'] = 1.0 if any(keyword.lower() == name.lower for keyword in [
        "AddCollateral",
        "AssetStatus",
        "CashAdded",
        "CashRemoved",
        "ClearedDeposit",
        "CollateralProvided",
        "CollateralUpdate",
        "CreateFNFT",
        "Deposit",
        "Deposited",
        "DepositToken",
        "EmergencyWithdraw",
        "FNFTAddionalDeposited",
        "FNFTAddressLockMinted",
        "FNFTWithdrawn",
        "GooBalanceUpdated",
        "InternalBalanceChanged",
        "Invest",
        "Lock",
        "Locked",
        "LogAddCollateral",
        "LogConvertedDeposit",
        "LogDeposit",
        "LogWithdraw",
        "OwnedAssetAdded",
        "PurchasePlp",
        "PlpDestroy",
        "Redeem",
        "RedeemFNFT",
        "RedeemUnderlying",
        "RemoveUnderlying",
        "Snapshot",
        "SupplyReceipt",
        "Sweep",
        "TokenBalanceChanged",
        "Unwrap",
        "VaultOpened",
        "VaultWithdraw",
        "Withdraw",
        "Withdrawal",
        "WithdrawAll",
        "WithdrawEvent",
        "Withdrawn",
        "WithdrawToken",
        "Wrap",
        "WrapNative"
      ]) else 0.0
        features['is_fees'] = 1.0 if any(keyword.lower() == name.lower for keyword in [
        "Fee",
        "FeeCollected",
        "FeePaid",
        "Fees",
        "FeesAccrued",
        "LUSDBorrowingFeePaid",
        "PerformanceFee",
        "RedeemFee",
        "ReferralCommissionPaid",
        "ReferralCommissionRecorded",
        "TakeSellFee",
        "TaxPayed",
        "VerifierFeePaid",
        "AssignBurnFees",
        "ExcludeFromFees",
        "ExcludeMultipleAccountsFromFees"
      ]) else 0.0
        features['is_governance'] = 1.0 if any(keyword.lower() == name.lower for keyword in [
        "DelegateChanged",
        "DelegatedPowerChanged",
        "DelegateVotesChanged",
        "InterfaceImplementerSet",
        "LogSetMasterContractApproval",
        "MetaTransactionExecuted",
        "Migrated",
        "migration",
        "NewEra",
        "NewWeight",
        "NodeAdded",
        "NotificationSent",
        "OwnershipTransferred",
        "ProposalExecuted",
        "SellQuotaChanged",
        "UpdateLiquidityLimit",
        "WhitelistedAddressAdded",
        "ChangedPartition",
        "Edit",
        "NewStaker",
        "TroveSnapshotsUpdated",
        "ExecuteTransaction",
        "UpdatePnl"
      ]) else 0.0
        features['is_cross-chain'] = 1.0 if any(keyword.lower() == name.lower for keyword in [
        "LiFiTransferStarted",
        "Packet",
        "RelayerParams",
        "SendToChain",
        "SentToFund",
        "TokensBridgingInitiated",
        "UserRequestForAffirmation",
        "In",
        "Received",
        "Send",
        "Sent"
      ]) else 0.0
        features['is_contract_state'] = 1.0 if any(keyword.lower() == name.lower for keyword in [
        "ContractFallbackCallFailed",
        "DidLCClose",
        "DidLCOpen",
        "Distribute",
        "Exercise",
        "Failure",
        "FinishedBet",
        "HandledDepeggedCurvePool",
        "LastFeeOpTimeUpdated",
        "OrderRecord",
        "onBuyBack",
        "onCredit",
        "ProcessBlockOverflow",
        "RoundFinished",
        "SafeReceived",
        "SteamGenerated",
        "SubscriptionFunded",
        "UpdatePositionFeeGrowthInside",
        "UpdatePositionUnclaimedFees",
        "UserState",
        "ValueDeletedUInt",
        "ValueSetUInt",
        "vGHSTLeft",
        "Work",
        "DecreaseGuaranteedUsd",
        "DecreaseReservedAmount",
        "DecreaseUsdgAmount",
        "DecreaseUsdpAmount",
        "IncreaseGuaranteedUsd",
        "IncreaseReservedAmount",
        "IncreaseUsdgAmount",
        "IncreaseUsdpAmount"
      ]) else 0.0


        # Parameter features
        param_count = name.count(',') + 1 if '(' in name else 0
        features['param_count'] = param_count / 10
        features['has_price'] = 1.0 if any(keyword in name for keyword in ['price', 'Price', 'value', 'Value']) else 0.0
        features['has_amount'] = 1.0 if any(keyword in name for keyword in ['amount', 'Amount', 'quantity']) else 0.0
        features['name_length'] = len(name) / 200
    else:
        features['is_transfer'] = 0.0
        features['is_approval'] = 0.0
        features['is_swap'] = 0.0
        features['is_mint_burn'] = 0.0
        features['param_count'] = 0.0
        features['has_price'] = 0.0
        features['has_amount'] = 0.0
        features['name_length'] = 0.0

        features['is_token_operations'] = 0.0
        features['is_liquidity'] = 0.0
        features['is_trading'] = 0.0
        features['is_lending'] = 0.0
        features['is_debt'] = 0.0
        features['is_oracle'] = 0.0
        features['is_staking'] = 0.0
        features['is_vaults'] = 0.0
        features['is_fees'] = 0.0
        features['is_governance'] = 0.0
        features['is_cross-chain'] = 0.0
        features['is_contract_state'] = 0.0


    # 4. Topics features
    MAX_TOPICS = 3
    if topics and topics != '':
        topic_list = [t.strip() for t in topics.split('\n') if t.strip()]
        features['topic_count'] = len(topic_list) / 10
        features['has_multiple_address'] = 1.0 if len(topic_list) >= 2 else 0.0
        features['is_circular_transfer'] = 1.0 if len(set(topic_list)) < len(topic_list) else 0.0  # Recurring transfers

        for i in range(MAX_TOPICS):
            if i < len(topic_list) and topic_list[i]:
                topic_hash = hashlib.sha256(topic_list[i].encode()).digest()
                features[f'topic{i + 1}_hash1'] = int.from_bytes(topic_hash[:4], 'big') / (2 ** 32 - 1)
                features[f'topic{i + 1}_hash2'] = int.from_bytes(topic_hash[4:8], 'big') / (2 ** 32 - 1)
            else:
                features[f'topic{i + 1}_hash1'] = 0.0
                features[f'topic{i + 1}_hash2'] = 0.0

    else:
        features['topic_count'] = 0.0
        features['has_multiple_address'] = 0.0
        features['is_circular_transfer'] = 0.0
        for i in range(MAX_TOPICS):
            features[f'topic{i + 1}_hash1'] = 0.0
            features[f'topic{i + 1}_hash2'] = 0.0


    # 5. Data features
    data_num = 0.0
    features['is_hex_address'] = 0.0
    features['hex_address_length'] = 0.0
    features['is_bool_value'] = 0.0
    features['bool_value'] = 0.0  
    features['data_type'] = 0.0  # 0 = Empty/Unknown, 1 = Number, 2 = Address, 3 = Boolean, 4 = Regular string
    features['data_addr_hash1'] = 0.0
    features['data_addr_hash2'] = 0.0
    features['data_value'] = 0.0
    features['is_zero_trade'] = -1.0
    features['is_negative'] = -1.0
    features['is_non_integer'] = -1.0
    features['is_abnormal_scale'] = -1.0
    features['is_large_trade'] = -1.0
    features['is_plain_string'] = 0.0  
    features['string_length'] = 0.0  
    features['string_has_special_char'] = 0.0  
    features['string_digit_ratio'] = 0.0  
    features['string_keyword_match'] = 0.0  
    features['data_str_hash1'] = 0.0  
    features['data_str_hash2'] = 0.0  

    if data and data != '':
        # Determine the boolean type.
        if data.lower() in ['true', 'false']:
            features['is_bool_value'] = 1.0
            features['bool_value'] = 1.0 if data.lower() == 'true' else 0.0
            features['data_type'] = 3.0
            features['data_value'] = features['bool_value']
            # Non-numeric/string-related features are set to zero.
            features['is_large_trade'] = 0.0
            features['is_zero_trade'] = 0.0
            features['is_negative'] = 0.0
            features['is_non_integer'] = 0.0
            features['is_abnormal_scale'] = 0.0
        # Determine the type of the 0x address.
        elif data.startswith('0x') and len(data) in [42, 66]: 
            features['is_hex_address'] = 1.0
            features['hex_address_length'] = len(data) / 100
            features['data_type'] = 2.0
            # Address hash features
            addr_hash = hashlib.sha256(data.encode()).digest()
            features['data_addr_hash1'] = int.from_bytes(addr_hash[:4], 'big') / (2 ** 32 - 1)
            features['data_addr_hash2'] = int.from_bytes(addr_hash[4:8], 'big') / (2 ** 32 - 1)
            # Basic numerical features (using address length as a substitute)
            features['data_value'] = len(data) / 100
            # Non-numeric/string-related features are set to zero.
            features['is_large_trade'] = 0.0
            features['is_zero_trade'] = 0.0
            features['is_negative'] = 0.0
            features['is_non_integer'] = 0.0
            features['is_abnormal_scale'] = 0.0
        # Determine the number type
        elif re.fullmatch(r'^-?\d+(\.\d+)?(e[+-]?\d+)?$', data, re.IGNORECASE):
            try:
                data_num = float(data)
            except ValueError:
                data_clean = re.sub(r'[^\d\.\-e]', '', str(data))
                data_num = float(data_clean) if data_clean else 0.0

            # Limit the range of `data_num` to prevent overflow after taking the logarithm.
            data_num_clipped = np.clip(data_num, -1e20, 1e20)
            # Calculate log1p and limit the maximum value.
            log_value = np.log1p(abs(data_num_clipped))
            features['data_value'] = min(log_value / 50, 1e5) 

            features['is_large_trade'] = 1.0 if abs(data_num_clipped) > 1e18 else 0.0
            features['is_zero_trade'] = 1.0 if data_num_clipped == 0 else 0.0
            features['is_negative'] = 1.0 if data_num_clipped < 0 else 0.0
            features['is_non_integer'] = 1.0 if '.' in str(data) else 0.0
            features['is_abnormal_scale'] = 1.0 if (
                    abs(data_num_clipped) > 0 and (
                    abs(data_num_clipped) < 1e6 or abs(data_num_clipped) > 1e20)) else 0.0
            features['data_type'] = 1.0

        # A sequence of characters that is not a boolean, address, or number, and has a length greater than 0.
        elif not (re.match(r'^-?\d+(\.\d+)?(e[+-]?\d+)?$', data, re.IGNORECASE)):
            features['is_plain_string'] = 1.0
            features['data_type'] = 4.0  # Marked as a regular string type.

            # String-specific feature: Length feature (normalized to the 0-1 range)
            str_len = len(data)
            features['string_length'] = min(str_len / 50, 1.0)  # The maximum length is limited to 50.

            # String-specific feature: Character type analysis
            total_chars = len(data)
            digit_count = sum(1 for c in data if c.isdigit())
            special_char_count = sum(1 for c in data if not (c.isalnum() or c.isspace()))

            features['string_digit_ratio'] = digit_count / total_chars if total_chars > 0 else 0.0
            # Does it contain special characters?
            features['string_has_special_char'] = 1.0 if special_char_count > 0 else 0.0

            # String-specific feature: Business keyword matching
            business_keywords = ['swap', 'log', 'burn', 'mint', 'approve', 'transfer', '3-0', '00']
            features['string_keyword_match'] = 1.0 if any(
                kw.lower() in data.lower() for kw in business_keywords) else 0.0

            # String-specific feature: Hashing features
            str_hash = hashlib.sha256(data.encode()).digest()
            features['data_str_hash1'] = int.from_bytes(str_hash[:4], 'big') / (2 ** 32 - 1)
            features['data_str_hash2'] = int.from_bytes(str_hash[4:8], 'big') / (2 ** 32 - 1)

            # String numerical feature (using the first 4 bytes of the hash as the base value)
            features['data_value'] = features['data_str_hash1']  # Reuse hash features as numerical values.

            features['is_large_trade'] = 0.0
            features['is_zero_trade'] = 0.0
            features['is_negative'] = 0.0
            features['is_non_integer'] = 0.0
            features['is_abnormal_scale'] = 0.0
        else:
            features['data_value'] = 0.0
            features['is_large_trade'] = 0.0
            features['is_zero_trade'] = 0.0
            features['is_negative'] = 0.0
            features['is_non_integer'] = 0.0
            features['is_abnormal_scale'] = 0.0
            features['data_type'] = 0.0
    else:
        features['data_value'] = 0.0
        features['is_large_trade'] = 0.0
        features['is_zero_trade'] = 1.0
        features['is_negative'] = 0.0
        features['is_non_integer'] = 0.0
        features['is_abnormal_scale'] = 0.0
        features['data_type'] = 0.0

    # ==================== Core price manipulation risk features====================
    # Attack pattern recognition
    features['is_manipulation_pattern'] = 1.0 if (
            features['is_transfer'] and
            (features['is_large_trade'] or features['is_abnormal_scale']) and
            features['has_multiple_address']
    ) else 0.0

    features['is_flash_loan_pattern'] = 1.0 if (
            features['is_swap'] and
            features['is_large_trade'] and
            not features['is_zero_trade']
    ) else 0.0

    features['is_wash_trading'] = 1.0 if (
            features['is_transfer'] and
            features['has_multiple_address'] and
            not features['is_large_trade'] and
            not features['is_zero_trade']
    ) else 0.0

    # The boolean type is associated with manipulation risks (such as True/False flags for scenarios like forced liquidation and settlement).
    features['bool_manipulation_risk'] = 1.0 if (
            features['is_bool_value'] and
            (features['bool_value'] == 1.0 and features['is_transfer'])
    ) else 0.0

    # Numerical anomaly Features
    features['price_manipulation_risk'] = calculate_price_manipulation_risk(data_num)
    features['amount_volatility'] = calculate_amount_volatility(data_num, historical_data)
    features['is_round_number'] = 1.0 if (
            data_num > 0 and str(data_num).replace('.', '').endswith('000')) else 0.0 

    # Address interaction error
    features['suspicious_address_trade'] = features['is_honey_pot_address'] * features['is_large_trade']
    features['contract_to_contract'] = features['is_contract_address'] * (
        1.0 if (topics and len(topic_list) >= 2 and all(len(t) == 42 for t in topic_list[:2])) else 0.0)

    # Time-series anomaly detection (requires historical data)
    if historical_data is not None and len(historical_data) > 0:
        features['trade_frequency'] = calculate_trade_frequency(number, historical_data)
    else:
        features['trade_frequency'] = 0.0

    # Multi-source risk portfolio scoring
    features['data_addr_match'] = 1.0 if (
            features['is_hex_address'] and
            address and
            data.lower() in address.lower()
    ) else 0.0

    # Combined risk Features
    features['high_risk_combination'] = sum([
        features['is_manipulation_pattern'] * 0.25,
        features['price_manipulation_risk'] * 0.2,
        features['suspicious_address_trade'] * 0.2,
        features['is_flash_loan_pattern'] * 0.15,
        features['addr_entropy'] * 0.1,
        features['bool_manipulation_risk'] * 0.05,  
        features['data_addr_match'] * 0.05 
    ])

    features['medium_risk_combination'] = sum([
        features['is_wash_trading'] * 0.4,
        features['amount_volatility'] * 0.3,
        features['contract_to_contract'] * 0.3
    ])

    # Features of a legally valid transaction
    features['is_standard_transfer'] = 1.0 if (
            features['is_transfer'] and
            not features['is_honey_pot_address'] and
            not features['is_abnormal_scale']
    ) else 0.0

    # Replace infinity and NaN values.
    for key, value in features.items():
        if np.isinf(value) or np.isnan(value):
            features[key] = 0.0

    features['risk_score'] = min(features['high_risk_combination'] + features['medium_risk_combination'], 1.0)

    # ==================== Global features ====================
    features['non_empty_fields'] = sum(1 for f in [number, address, name, topics, data] if f and f != '') / 4
    features['total_text_length'] = (len(str(address)) + len(str(name)) + len(str(topics))) / 500

    # ==================== Convert to feature vectors ====================
    feature_names = sorted(features.keys())
    feature_vector = np.array([features[name] for name in feature_names], dtype=np.float32)

    return feature_vector, feature_names


# ==================== Price manipulation attack functions ====================
def calculate_address_entropy(address):
    """Calculate address entropy"""
    if not address or len(address) < 10:
        return 0.0
    address_clean = address.replace('0x', '')
    chars = list(address_clean)
    char_counts = {}
    for c in chars:
        char_counts[c] = char_counts.get(c, 0) + 1
    entropy = 0.0
    total = len(chars)
    for count in char_counts.values():
        p = count / total
        entropy -= p * np.log2(p + 1e-8)
    return entropy / np.log2(16)  


def calculate_price_manipulation_risk(amount):
    """Calculate the price manipulation risk score."""
    if amount == 0:
        return 0.0
    abs_amount = abs(amount)
    risk = 0.0
    if abs_amount > 1e20 or abs_amount < 1e6:
        risk += 0.4
    if not str(amount).replace('.', '').isdigit():
        risk += 0.3
    if abs_amount > 1e18:
        risk += 0.2
    if str(abs_amount).count('9') > 3:
        risk += 0.1
    return min(risk, 1.0)


def calculate_amount_volatility(amount, historical_data):
    """Calculate the volatility of the amount (requires historical data)"""
    if historical_data is None or len(historical_data) < 5 or amount == 0:
        return 0.0
    historical_amounts = [float(d.get('Data', 0)) for d in historical_data if
                          d.get('Data', '0').replace('.', '').isdigit()]
    if len(historical_amounts) < 2:
        return 0.0
    mean = np.mean(historical_amounts)
    std = np.std(historical_amounts)
    if std == 0:
        return 0.0
    z_score = abs(amount - mean) / std
    return min(z_score / 3, 1.0) 


def calculate_trade_frequency(current_number, historical_data):
    """Calculate transaction frequency anomalies."""
    if historical_data is None or len(historical_data) < 10:
        return 0.0
    recent_trades = [d for d in historical_data if int(d.get('Number', 0)) > current_number - 10]
    frequency = len(recent_trades) / 10
    return min(frequency, 1.0)



def create_price_manipulation_256d_embedding(records, historical_data=None, random_state=42):
    """
    Generate a 256-dimensional embedding specifically for detecting price manipulation attacks.
    """
    # Extract all features.
    feature_vectors = []
    feature_names = None

    for record in records:
        vec, names = price_manipulation_feature_engineering(record, historical_data)
        if feature_names is None:
            feature_names = names
        feature_vectors.append(vec)

    feature_matrix = np.array(feature_vectors, dtype=np.float32)
    n_samples, n_features = feature_matrix.shape

    print(f"Number of samples: {n_samples}, Feature dimension: {n_features}")

    # Data cleaning (handling infinite and NaN values)
    feature_matrix = np.nan_to_num(
        feature_matrix,
        nan=0.0,
        posinf=np.finfo(np.float32).max,
        neginf=np.finfo(np.float32).min
    )
    # Limit the numerical range to within the safe range of float32.
    feature_matrix = np.clip(feature_matrix, -1e18, 1e18)

    # Preprocessing
    imputer = SimpleImputer(strategy='constant', fill_value=0.0)
    feature_matrix_imputed = imputer.fit_transform(feature_matrix)

    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix_imputed)

    # PCA logic
    if n_samples <= 1:
        embedding_256d = np.zeros((1, 256), dtype=np.float32)
        embedding_256d[0, :n_features] = feature_matrix_scaled[0]

    elif n_features < 256:
        # PCA adapted for small sample sizes - using randomized SVD
        max_pca_components = min(n_samples - 1, n_features)  
        if max_pca_components < 1:
            max_pca_components = 1

        # Using randomized SVD to avoid the problem of insufficient sample size.
        pca_full = PCA(
            n_components=max_pca_components,
            random_state=random_state,
            svd_solver='randomized'  # randomized solver
        )
        pca_features = pca_full.fit_transform(feature_matrix_scaled)

        # Generate additional dimensions
        np.random.seed(random_state)
        additional_dims = []
        dim_to_add = 256 - max_pca_components

        # Index of Core Characteristics of Price Manipulation
        manipulation_feature_indices = []
        for i, name in enumerate(feature_names):
            if i < max_pca_components and any(
                    keyword in name for keyword in ['manipulation', 'risk', 'flash', 'wash', 'suspicious']):
                manipulation_feature_indices.append(i)

        # Generate additional dimensions
        for _ in range(dim_to_add):
            if len(manipulation_feature_indices) >= 2:
                # Prioritizing combinations of price manipulation characteristics
                idx1, idx2 = np.random.choice(manipulation_feature_indices, 2, replace=True)
                combo = (pca_features[:, idx1] * 0.7 + pca_features[:, idx2] * 0.3)
                combo = combo * (0.9 + 0.2 * np.random.rand(n_samples))
            elif max_pca_components >= 2:
                # Randomly combine existing features.
                idx1, idx2 = np.random.choice(max_pca_components, 2, replace=True)
                combo = (pca_features[:, idx1] + pca_features[:, idx2]) / 2
            else:
                # Only 1-dimensional features; adding noisy variations.
                combo = pca_features[:, 0] * (0.8 + 0.4 * np.random.rand(n_samples))

            additional_dims.append(combo)

        # Merge features
        additional_dims = np.array(additional_dims, dtype=np.float32).T
        embedding_256d = np.hstack([pca_features, additional_dims])

        # Final standardization
        embedding_scaler = StandardScaler()
        embedding_256d = embedding_scaler.fit_transform(embedding_256d)

    elif n_features == 256:
        embedding_256d = feature_matrix_scaled

    else:
        # Reduced to 256 dimensions.
        max_pca_components = min(n_samples - 1, 256)
        if max_pca_components < 1:
            max_pca_components = 1

        pca = PCA(
            n_components=max_pca_components,
            random_state=random_state,
            svd_solver='randomized'
        )
        pca_features = pca.fit_transform(feature_matrix_scaled)

        # Padded to 256 dimensions.
        additional_dims = np.random.randn(n_samples, 256 - max_pca_components) * 0.01
        embedding_256d = np.hstack([pca_features, additional_dims])

        print(f"PCA dimensionality reduction retains a certain proportion of the variance.: {np.sum(pca.explained_variance_ratio_):.4f}")

    # Risk features weighting
    embedding_256d = weight_manipulation_features(embedding_256d, feature_names)

    embedding_256d = np.nan_to_num(embedding_256d, nan=0.0, posinf=1.0, neginf=-1.0)
    # Avoid division by zero.
    norms = np.linalg.norm(embedding_256d, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    embedding_256d = embedding_256d / norms

    print(f"Final embedded shape: {embedding_256d.shape}")

    return embedding_256d, feature_names


def weight_manipulation_features(embedding, feature_names):
    """Weighting the features related to price manipulation."""
    n_actual_dims = min(len(feature_names), embedding.shape[1])

    # Identifying dimensions related to price manipulation
    manipulation_dim_indices = []
    for i, name in enumerate(feature_names[:n_actual_dims]):
        if any(keyword in name for keyword in ['manipulation', 'risk', 'flash', 'wash', 'suspicious']):
            manipulation_dim_indices.append(i)

    # Weighting these dimensions
    if manipulation_dim_indices:
        embedding[:, manipulation_dim_indices] = embedding[:, manipulation_dim_indices] * 1.5

    return embedding


def calculate_manipulation_risk_scores(embedding):
    """Calculate the price manipulation risk score."""
    # Extracting risk-related dimensions
    risk_dims = embedding[:, :min(50, embedding.shape[1])]

    # Calculate the risk score
    risk_scores = np.mean(risk_dims, axis=1)
    risk_scores = (risk_scores - risk_scores.min()) / (risk_scores.max() - risk_scores.min() + 1e-8)

    return risk_scores


# ==================== main ====================
if __name__ == "__main__":

    dataset_name = ['attack incident', 'high_value'] #'attack incident', 'high_value'
    # dataset_name = ['high_value_full']
    platform = ['ARB', 'AVAX', 'Base', 'BSC', 'ETH', 'POL']
    error = []
    for dn in dataset_name:
        for pf in platform:
            protocol_path = "../../dataset/" + dn + "/" + pf + "/"
            if os.path.exists(protocol_path):
                protocol_list = os.listdir(protocol_path)
                for pt in protocol_list:
                    start = time.time()
                    if os.path.exists('../embeddings/' + dn + '/' + pf + '/' + pt + '.csv'):
                        print(pt + " The embedding already exists.")
                        continue

                    print(pf + ' + ' + pt)
                    EXCEL_FILE_PATH = protocol_path + pt + "/Event.xlsx"
                    try:
                        df = pd.read_excel(EXCEL_FILE_PATH, converters={'Data': str}).fillna('')
                        # Ensure that you only select the necessary columns.
                        if all(col in df.columns for col in ['Number', 'Address', 'Name', 'Topics', 'Data']):
                            records = [tuple(row) for _, row in df[['Number', 'Address', 'Name', 'Topics', 'Data']].iterrows()]
                        else:
                            print("Error: Excel column names do not match.")
                            traceback.print_exc()
                    except Exception as e:
                        print(f"Failed to load Excel file: {e}")
                        traceback.print_exc()

                    # Preparing historical data
                    historical_data = df.to_dict('records') 

                    # Generate 256-dimensional embeddings specifically for price manipulation attacks.
                    try:
                        embedding_256d, feature_names = create_price_manipulation_256d_embedding(records, historical_data)

                        embedding_columns = [f'embedding_dim_{i}' for i in range(embedding_256d.shape[1])]
                        embedding_df = pd.DataFrame(embedding_256d, columns=embedding_columns)
                        OUTPUT_PATH = '../embeddings/' + dn + '/' + pf + '/'

                        if not os.path.exists(OUTPUT_PATH):
                            os.makedirs(OUTPUT_PATH)
                        embedding_df.to_csv(OUTPUT_PATH + pt + ".csv", index=False)
                        print("\nEmbedded content has been saved.: " + OUTPUT_PATH + pt + ".csv")

                        # Risk score
                        """
                        risk_scores = calculate_manipulation_risk_scores(embedding_256d)
                        print(f"\\n Price manipulation risk score (0-1, higher score indicates higher risk):")
                        for i, score in enumerate(risk_scores[:10]):  
                            risk_level = "High risk if score > 0.7 else  Medium risk if score > 0.3 else Low risk"
                            print(f"Record {i + 1}: {score:.4f} - {risk_level}")
                        """

                    except Exception as e:
                        print(f"An error occurred while generating the embedding.: {e}")
                        error.append(pt)
                        # traceback.print_exc()
                    end = time.time()


    print(error)
