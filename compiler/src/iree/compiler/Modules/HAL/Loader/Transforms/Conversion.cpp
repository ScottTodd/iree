// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Conversion/StandardToHAL/Patterns.h"
#include "iree/compiler/Dialect/HAL/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/HAL/Conversion/UtilToHAL/Patterns.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/Util/Conversion/ConversionPatterns.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Modules/HAL/Inline/Conversion/HALToHALInline/Patterns.h"
#include "iree/compiler/Modules/HAL/Inline/Conversion/StreamToHALInline/Patterns.h"
#include "iree/compiler/Modules/HAL/Inline/IR/HALInlineDialect.h"
#include "iree/compiler/Modules/HAL/Loader/Conversion/StreamToHALLoader/Patterns.h"
#include "iree/compiler/Modules/HAL/Loader/IR/HALLoaderDialect.h"
#include "iree/compiler/Modules/HAL/Loader/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler::IREE::HAL::Loader {

#define GEN_PASS_DEF_CONVERSIONPASS
#include "iree/compiler/Modules/HAL/Loader/Transforms/Passes.h.inc"

namespace {

// Runs conversion with registered input dialects.
class ConversionPass final : public impl::ConversionPassBase<ConversionPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Util::UtilDialect, IREE::HAL::HALDialect,
                    IREE::HAL::Inline::HALInlineDialect,
                    IREE::HAL::Loader::HALLoaderDialect,
                    mlir::arith::ArithDialect, mlir::affine::AffineDialect>();
  }

  void runOnOperation() override {
    auto *context = &getContext();

    // Ensure all input dialects go away.
    ConversionTarget conversionTarget(*context);
    conversionTarget.addLegalDialect<
        mlir::func::FuncDialect, mlir::scf::SCFDialect,
        mlir::arith::ArithDialect, mlir::affine::AffineDialect>();

    SmallVector<const HALConversionDialectInterface *> conversionInterfaces;
    for (auto *dialect : context->getLoadedDialects()) {
      if (auto *conversionInterface =
              dialect
                  ->getRegisteredInterface<HALConversionDialectInterface>()) {
        conversionInterfaces.emplace_back(conversionInterface);
      }
    }

    HALTypeConverter typeConverter(conversionInterfaces);
    RewritePatternSet patterns(context);

    // Pass-through.
    typeConverter.addConversion([](IndexType type) { return type; });
    typeConverter.addConversion([](IntegerType type) { return type; });
    typeConverter.addConversion([](FloatType type) { return type; });
    typeConverter.addConversion(
        [](IREE::Util::BufferType type) { return type; });

    // Convert stream into `hal_inline`, which mostly entails ignoring ops.
    // We override those related to executables to `hal_loader` by way of high
    // pattern benefits.
    conversionTarget.addLegalDialect<IREE::HAL::Inline::HALInlineDialect>();
    populateStreamToHALInlinePatterns(context, conversionTarget, typeConverter,
                                      patterns);
    conversionTarget.addLegalDialect<IREE::HAL::Loader::HALLoaderDialect>();
    populateStreamToHALLoaderPatterns(context, conversionTarget, typeConverter,
                                      patterns);

    // Convert some common things into HAL, reusing those conversions.
    populateUtilToHALPatterns(context, conversionTarget, typeConverter,
                              patterns);
    populateStandardToHALPatterns(context, conversionTarget, typeConverter,
                                  patterns);

    // Convert any full `hal` ops into `hal_inline` ops.
    conversionTarget.addIllegalDialect<IREE::HAL::HALDialect>();
    populateHALToHALInlinePatterns(context, conversionTarget, typeConverter,
                                   patterns);

    // Generic conversion.
    conversionTarget.addLegalDialect<IREE::Util::UtilDialect>();
    populateUtilConversionPatterns(context, conversionTarget, typeConverter,
                                   patterns);

    if (failed(applyPartialConversion(getOperation(), conversionTarget,
                                      std::move(patterns)))) {
      getOperation().emitError()
          << "conversion to the hal_inline + hal_loader dialects failed";
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL::Loader
