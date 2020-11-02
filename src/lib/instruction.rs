//! The instruction set

use crate::typeinfo::{ TypeID, TypeKind };

/// An enum over every instruction type
#[repr(C, u8)]
#[allow(missing_docs)]
pub enum Instruction {
  // Variable access //

  LoadLocal(u8),
  StoreLocal(u8),

  LoadGlobal(u16),
  LoadGlobalDeferred,
  StoreGlobal(u16),
  StoreGlobalDeferred,


  // Collection access //

  GetField(u8),
  GetFieldDeferred,
  SetField(u8),
  SetFieldDeferred,

  GetArrayElement,
  SetArrayElement,

  GetMapElement,
  SetMapElement,

  GetStringByte,
  GetStringCodepoint,
  SetStringByte,
  SetStringCodepoint,


  // Constructors //

  CreateRecord(TypeID),
  CreateArray(TypeID),
  CreateMap(TypeID),
  CreateString,


  // Constant values //

  ConstReal(f64),
  ConstInteger(i32),
  ConstCharacter(char),
  ConstBoolean(bool),
  ConstNil,
  ConstTypeID(TypeID),
  ConstString(String),


  // Unary ops //

  NegateReal,
  NegateInteger,

  AbsReal,
  AbsInteger,

  NotInteger,
  NotBoolean,


  // Binary ops //

  AddReal,
  AddInteger,

  SubReal,
  SubInteger,

  MulReal,
  MulInteger,

  DivReal,
  DivInteger,

  RemReal,
  RemInteger,

  PowReal,
  PowInteger,

  AndInteger,
  OrInteger,
  XorInteger,

  AndBoolean,
  OrBoolean,

  EqReal,
  EqInteger,
  EqCharacter,
  EqBoolean,
  
  NeReal,
  NeInteger,
  NeCharacter,
  NeBoolean,

  GtReal,
  GtInteger,
  GtCharacter,

  LtReal,
  LtInteger,
  LtCharacter,

  GeReal,
  GeInteger,
  GeCharacter,

  LeReal,
  LeInteger,
  LeCharacter,


  // Control flow //

  Call,

  Branch(usize),
  ConditionalBranch(usize, usize),

  Return,

  Panic,


  // Conversion //

  CastRealToInteger,
  CastIntegerToReal,
  CastCharacterToInteger,
  CastIntegerToCharacter,
  CastIntegerToBoolean,
  CastBooleanToInteger,
  CastDeferred,


  // Introspection //

  TypeIDOfLocal(u8),
  TypeIDOfGlobal(u16),
  TypeIDOfGlobalDeferred,
  TypeIDOfField(u8),
  TypeIDOfFieldDeferred,
  TypeIDOfKey,
  TypeIDOfElement,
  TypeIDOfParameter(u8),
  TypeIDOfReturn,
  TypeIDOfValue,
  
  GlobalExists,
  FieldExists,
  ParameterExists,
  ReturnExists,

  TypeIDToName,

  FieldCount,
  ElementCount,
  ParameterCount,

  IsTypeKind(TypeKind),
  
  EqTypeID,
  NeTypeID,
}